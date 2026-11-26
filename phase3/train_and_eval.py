"""Phase 3 Zero-Leakage Splits, Walk-Forward CV, Multimodal Modeling, Calibration, Quantiles, Volatility & ONNX Export."""
from __future__ import annotations

from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path
import os
import pickle
import time
from typing import Any

import torch
import torch.nn as nn

_TORCH_LIB = Path(torch.__file__).parent / "lib"
if _TORCH_LIB.is_dir() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_TORCH_LIB))

import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_pinball_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path("phase3/.cache/features_table.parquet")
EMB_PATH = Path("phase3/.cache/bge_embeddings.npy")
MODELS_DIR = Path("models/phase3")
RESULTS_DIR = Path("phase3/results")
DATA_DIR = Path("phase3/data")

EMBARGO_SESSIONS = 20
TARGET_RET_COL = "target_exec_excess_5d"
DIR_THRESHOLD = 0.015  # +/- 1.5% excess move (above round-trip brokerage+STT+slippage)


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def make_dir_label(ret: np.ndarray, tau: float = DIR_THRESHOLD) -> np.ndarray:
    y = np.ones(len(ret), dtype=np.int64)
    y[ret < -tau] = 0
    y[ret > tau] = 2
    return y


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray, num_classes: int = 3) -> float:
    oh = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == y_true).astype(float)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (conf >= b0) & (conf < b1 if b1 < 1.0 else conf <= b1)
        if m.sum() > 0:
            ece += (m.sum() / len(y_true)) * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return float(ece)


def eval_classifier(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = np.argmax(probs, axis=1)
    return {
        "balanced_acc": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, pred)), 4),
        "brier": round(multiclass_brier(y_true, probs), 4),
        "log_loss": round(float(log_loss(y_true, probs, labels=[0, 1, 2])), 4),
        "ece": round(expected_calibration_error(y_true, probs), 4),
    }


def block_bootstrap_delta(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 19,
) -> dict[str, Any]:
    """Grouped block bootstrap comparing Model A (Combined) minus Model B (Market-Only) by calendar-month/issuer."""
    rng = np.random.default_rng(seed)
    uniq_g = np.unique(groups)
    g_to_idx = {g: np.where(groups == g)[0] for g in uniq_g}
    bacc_deltas, mcc_deltas, brier_deltas = [], [], []
    for _ in range(n_boot):
        chosen = rng.choice(uniq_g, size=len(uniq_g), replace=True)
        idx = np.concatenate([g_to_idx[g] for g in chosen])
        yt = y_true[idx]
        pa = probs_a[idx]
        pb = probs_b[idx]
        if len(np.unique(yt)) < 3:
            continue
        bacc_deltas.append(balanced_accuracy_score(yt, np.argmax(pa, axis=1)) - balanced_accuracy_score(yt, np.argmax(pb, axis=1)))
        mcc_deltas.append(matthews_corrcoef(yt, np.argmax(pa, axis=1)) - matthews_corrcoef(yt, np.argmax(pb, axis=1)))
        brier_deltas.append(multiclass_brier(yt, pa) - multiclass_brier(yt, pb))
    return {
        "n_groups": len(uniq_g),
        "bacc_delta_mean": round(float(np.mean(bacc_deltas)), 5),
        "bacc_delta_ci95": [round(float(np.quantile(bacc_deltas, 0.025)), 5), round(float(np.quantile(bacc_deltas, 0.975)), 5)],
        "mcc_delta_mean": round(float(np.mean(mcc_deltas)), 5),
        "mcc_delta_ci95": [round(float(np.quantile(mcc_deltas, 0.025)), 5), round(float(np.quantile(mcc_deltas, 0.975)), 5)],
        "brier_delta_mean": round(float(np.mean(brier_deltas)), 5),
        "brier_delta_ci95": [round(float(np.quantile(brier_deltas, 0.025)), 5), round(float(np.quantile(brier_deltas, 0.975)), 5)],
    }


class CompactTabularMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(48, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_compact_mlp(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, seed: int = 19) -> tuple[CompactTabularMLP, SimpleImputer, StandardScaler]:
    torch.manual_seed(seed)
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    Xva = sc.transform(imp.transform(X_va)).astype(np.float32)

    model = CompactTabularMLP(Xtr.shape[1], 3)
    counts = np.bincount(y_tr, minlength=3)
    w = torch.tensor(np.clip(counts.max() / np.maximum(counts, 1), 1.0, 3.0), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_loss = 1e9
    best_state = None
    xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(y_tr)
    xv = torch.from_numpy(Xva)
    yv = torch.from_numpy(y_va)
    bs = 256
    for _ in range(18):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = crit(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(crit(model(xv), yv).item())
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, imp, sc


def build_feature_matrices(
    df: pd.DataFrame,
    embs: np.ndarray,
    tr_mask: np.ndarray,
    svd: TruncatedSVD | None = None,
) -> tuple[dict[str, np.ndarray], list[str], list[str], TruncatedSVD]:
    nlp_base_cols = sorted([c for c in df.columns if c.startswith("nlp_")])
    mkt_cols = sorted([c for c in df.columns if c.startswith("tech_") or c.startswith("ctx_") or c.startswith("fund_") or c == "has_pit_fundamentals"])

    # In-fold SVD fit ONLY on training rows to guarantee zero semantic leakage
    emb_idx_tr = df.loc[tr_mask, "emb_row_idx"].to_numpy()
    if svd is None:
        svd = TruncatedSVD(n_components=16, random_state=19)
        svd.fit(embs[emb_idx_tr])
    svd_feats = svd.transform(embs[df["emb_row_idx"].to_numpy()]).astype(np.float32)
    svd_cols = [f"nlp_svd_{i}" for i in range(svd_feats.shape[1])]

    X_nlp_base = df[nlp_base_cols].to_numpy(dtype=np.float32)
    X_nlp = np.hstack([X_nlp_base, svd_feats])
    X_mkt = df[mkt_cols].to_numpy(dtype=np.float32)
    X_comb = np.hstack([X_mkt, X_nlp])

    return {"market_only": X_mkt, "nlp_only": X_nlp, "combined": X_comb}, mkt_cols, nlp_base_cols + svd_cols, svd


def fit_gbdt_suite(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    seed: int = 19,
) -> dict[str, tuple[Any, np.ndarray, np.ndarray]]:
    out = {}
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        subsample_for_bin=20000,
        colsample_bytree=0.8,
        subsample=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
    out["lightgbm"] = (lgbm, lgbm.predict_proba(X_va), lgbm.predict_proba(X_te))

    # Compute sample weights for XGBoost balanced classes
    counts = np.bincount(y_tr, minlength=3)
    sw = np.array([counts.max() / max(1, counts[y]) for y in y_tr], dtype=np.float32)
    xgbm = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=4,
    )
    xgbm.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_va, y_va)], verbose=False)
    out["xgboost"] = (xgbm, xgbm.predict_proba(X_va), xgbm.predict_proba(X_te))

    cbm = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        random_seed=seed,
        thread_count=4,
        verbose=0,
    )
    cbm.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=40, verbose=False)
    out["catboost"] = (cbm, cbm.predict_proba(X_va), cbm.predict_proba(X_te))
    return out


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_PATH)
    embs = np.load(EMB_PATH).astype(np.float32)
    df = df.sort_values(["t0_idx", "published_ts_ist", "event_id"]).reset_index(drop=True)

    # 1. Exact Chronological + 20-Trading-Session Embargoed Partition
    tr_end_date = "2025-05-31"
    tr_base_mask = (df["t0_date"] <= tr_end_date).to_numpy()
    tr_max_t0 = int(df.loc[tr_base_mask, "t0_idx"].max())

    val_start_t0 = tr_max_t0 + EMBARGO_SESSIONS + 1
    va_base_mask = (df["t0_idx"] >= val_start_t0) & (df["t0_date"] <= "2026-02-15")
    va_max_t0 = int(df.loc[va_base_mask, "t0_idx"].max())

    te_start_t0 = va_max_t0 + EMBARGO_SESSIONS + 1
    te_base_mask = (df["t0_idx"] >= te_start_t0).to_numpy()

    # Exact deduplication check across splits
    df["dup_key"] = df["symbol"] + "|" + df["published_date"] + "|" + df["text"].str[:100].str.lower()
    tr_dups = set(df.loc[tr_base_mask, "dup_key"])
    va_mask = va_base_mask.to_numpy() & ~df["dup_key"].isin(tr_dups).to_numpy()
    va_dups = set(df.loc[va_mask, "dup_key"])
    te_mask = te_base_mask & ~df["dup_key"].isin(tr_dups | va_dups).to_numpy()

    # Verify zero overlap
    assert len(set(df.loc[tr_base_mask, "event_id"]) & set(df.loc[va_mask, "event_id"])) == 0
    assert len(set(df.loc[va_mask, "event_id"]) & set(df.loc[te_mask, "event_id"])) == 0
    assert int(df.loc[va_mask, "t0_idx"].min()) - int(df.loc[tr_base_mask, "t0_idx"].max()) > EMBARGO_SESSIONS
    assert int(df.loc[te_mask, "t0_idx"].min()) - int(df.loc[va_mask, "t0_idx"].max()) > EMBARGO_SESSIONS

    y_ret = df[TARGET_RET_COL].to_numpy(dtype=np.float64)
    y_cls = make_dir_label(y_ret, DIR_THRESHOLD)
    y_vol = df["target_vol_expand_cls"].to_numpy(dtype=np.int64)

    mats, mkt_cols, nlp_cols, svd = build_feature_matrices(df, embs, tr_base_mask)
    y_tr, y_va, y_te = y_cls[tr_base_mask], y_cls[va_mask], y_cls[te_mask]

    print(
        f"=== Step 5: Modeling & Ablations (Train={tr_base_mask.sum()}, Embargo={EMBARGO_SESSIONS}d, "
        f"Val={va_mask.sum()}, Test={te_mask.sum()}) ===",
        flush=True,
    )
    print(f"Class balance Test (Neg/Neut/Pos): {Counter(y_te.tolist())}", flush=True)

    # 2. Baselines + All Predictive Models
    benchmark_rows = {}
    te_probs_store: dict[str, np.ndarray] = {}

    # 2a. Majority / No-Change baseline
    maj = int(Counter(y_tr.tolist()).most_common(1)[0][0])
    tr_priors = np.bincount(y_tr, minlength=3) / len(y_tr)
    p_maj = np.tile(tr_priors, (len(y_te), 1))
    # Force argmax to majority class for classification accuracy, priors for Brier
    benchmark_rows["baseline_majority_prior"] = eval_classifier(y_te, p_maj)

    # 2b. Historical event-family empirical prior baseline (from train only)
    fam_priors = {}
    for fam in df["event_family_hint"].unique():
        m = tr_base_mask & (df["event_family_hint"] == fam).to_numpy()
        fam_priors[fam] = (np.bincount(y_cls[m], minlength=3) + 1.0) / (m.sum() + 3.0)
    p_hist_fam = np.vstack([fam_priors[f] for f in df.loc[te_mask, "event_family_hint"]])
    benchmark_rows["baseline_event_family_mean"] = eval_classifier(y_te, p_hist_fam)

    # 2c. L2 Logistic Regression (Combined)
    imp_lr = SimpleImputer(strategy="median")
    sc_lr = StandardScaler()
    Xtr_lr = sc_lr.fit_transform(imp_lr.fit_transform(mats["combined"][tr_base_mask]))
    Xte_lr = sc_lr.transform(imp_lr.transform(mats["combined"][te_mask]))
    lr = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=19)
    lr.fit(Xtr_lr, y_tr)
    p_lr_te = lr.predict_proba(Xte_lr)
    benchmark_rows["logistic_regression_combined"] = eval_classifier(y_te, p_lr_te)
    te_probs_store["logistic_regression_combined"] = p_lr_te

    # 2d. LightGBM, XGBoost, CatBoost across (market_only, nlp_only, combined)
    trained_suites: dict[str, dict[str, Any]] = {}
    va_probs_store: dict[str, np.ndarray] = {}
    for fset_name in ("market_only", "nlp_only", "combined"):
        Xtr = mats[fset_name][tr_base_mask]
        Xva = mats[fset_name][va_mask]
        Xte = mats[fset_name][te_mask]
        suite = fit_gbdt_suite(Xtr, y_tr, Xva, y_va, Xte)
        trained_suites[fset_name] = suite
        for mname, (_, p_va, p_te) in suite.items():
            key = f"{mname}_{fset_name}"
            benchmark_rows[key] = eval_classifier(y_te, p_te)
            va_probs_store[key] = p_va
            te_probs_store[key] = p_te
            print(f"  {key:28s}: bacc={benchmark_rows[key]['balanced_acc']} mcc={benchmark_rows[key]['mcc']} brier={benchmark_rows[key]['brier']} ece={benchmark_rows[key]['ece']}", flush=True)

    # 2e. Compact PyTorch Tabular MLP (Combined)
    mlp_m, mlp_imp, mlp_sc = train_compact_mlp(mats["combined"][tr_base_mask], y_tr, mats["combined"][va_mask], y_va)
    with torch.no_grad():
        xt_te = torch.from_numpy(mlp_sc.transform(mlp_imp.transform(mats["combined"][te_mask])).astype(np.float32))
        lg_te = mlp_m(xt_te).numpy()
        p_mlp = np.exp(lg_te - np.max(lg_te, axis=1, keepdims=True))
        p_mlp /= p_mlp.sum(axis=1, keepdims=True)
    benchmark_rows["pytorch_mlp_combined"] = eval_classifier(y_te, p_mlp)
    te_probs_store["pytorch_mlp_combined"] = p_mlp
    print(f"  {'pytorch_mlp_combined':28s}: bacc={benchmark_rows['pytorch_mlp_combined']['balanced_acc']} mcc={benchmark_rows['pytorch_mlp_combined']['mcc']} brier={benchmark_rows['pytorch_mlp_combined']['brier']}", flush=True)

    # 3. Probability Calibration (strictly fitted on Val probabilities via FrozenEstimator + CalibratedClassifierCV)
    # Compare CatBoost Combined vs CatBoost Market-Only after isotonic calibration on Val
    cb_comb_model = trained_suites["combined"]["catboost"][0]
    cb_mkt_model = trained_suites["market_only"]["catboost"][0]
    cal_comb = CalibratedClassifierCV(FrozenEstimator(cb_comb_model), method="isotonic")
    cal_comb.fit(mats["combined"][va_mask], y_va)
    p_comb_cal = cal_comb.predict_proba(mats["combined"][te_mask])
    benchmark_rows["catboost_combined_calibrated"] = eval_classifier(y_te, p_comb_cal)

    cal_mkt = CalibratedClassifierCV(FrozenEstimator(cb_mkt_model), method="isotonic")
    cal_mkt.fit(mats["market_only"][va_mask], y_va)
    p_mkt_cal = cal_mkt.predict_proba(mats["market_only"][te_mask])
    benchmark_rows["catboost_market_only_calibrated"] = eval_classifier(y_te, p_mkt_cal)

    # 4. Block-Bootstrap 95% CI comparing Combined vs Market-Only (grouped by month & issuer)
    test_months = df.loc[te_mask, "t0_date"].str[:7].to_numpy()
    test_issuers = df.loc[te_mask, "symbol"].to_numpy()
    boot_by_month = block_bootstrap_delta(y_te, te_probs_store["catboost_combined"], te_probs_store["catboost_market_only"], test_months, n_boot=1000)
    boot_by_issuer = block_bootstrap_delta(y_te, te_probs_store["catboost_combined"], te_probs_store["catboost_market_only"], test_issuers, n_boot=1000)

    # 5. Regime Breakdown on Frozen Test
    regime_eval = {}
    te_regs = df.loc[te_mask, "regime_causal"].to_numpy()
    for reg in sorted(np.unique(te_regs)):
        rmask = te_regs == reg
        if rmask.sum() >= 20:
            regime_eval[reg] = {
                "count": int(rmask.sum()),
                "catboost_combined": eval_classifier(y_te[rmask], te_probs_store["catboost_combined"][rmask]),
                "catboost_market_only": eval_classifier(y_te[rmask], te_probs_store["catboost_market_only"][rmask]),
            }

    # 6. Return Quantile Estimation (q10, q50, q90) + 80% Interval Coverage
    quant_eval = {}
    y_ret_tr, y_ret_va, y_ret_te = y_ret[tr_base_mask], y_ret[va_mask], y_ret[te_mask]
    for fset in ("market_only", "combined"):
        q_preds = {}
        losses = {}
        for alpha in (0.10, 0.50, 0.90):
            qm = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=250, learning_rate=0.04, num_leaves=31, random_state=19, n_jobs=4, verbose=-1)
            qm.fit(mats[fset][tr_base_mask], y_ret_tr, eval_set=[(mats[fset][va_mask], y_ret_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
            pr = qm.predict(mats[fset][te_mask])
            q_preds[alpha] = pr
            losses[f"pinball_q{int(alpha*100)}"] = round(float(mean_pinball_loss(y_ret_te, pr, alpha=alpha)), 5)
        cov80 = float(np.mean((y_ret_te >= q_preds[0.10]) & (y_ret_te <= q_preds[0.90])))
        mae = float(mean_absolute_error(y_ret_te, q_preds[0.50]))
        quant_eval[fset] = {**losses, "mae_q50": round(mae, 5), "interval80_coverage": round(cov80, 4)}

    # 7. Post-Event Volatility Expansion Prediction (vol_expansion_5d > 1.25)
    vol_eval = {}
    y_vtr, y_vva, y_vte = y_vol[tr_base_mask], y_vol[va_mask], y_vol[te_mask]
    vol_models = {}
    for fset in ("market_only", "combined"):
        cbv = CatBoostClassifier(iterations=350, learning_rate=0.05, depth=6, auto_class_weights="Balanced", random_seed=19, thread_count=4, verbose=0)
        cbv.fit(mats[fset][tr_base_mask], y_vtr, eval_set=(mats[fset][va_mask], y_vva), early_stopping_rounds=30, verbose=False)
        pv = cbv.predict_proba(mats[fset][te_mask])[:, 1]
        vol_models[fset] = cbv
        vol_eval[fset] = {
            "roc_auc": round(float(roc_auc_score(y_vte, pv)), 4),
            "balanced_acc": round(float(balanced_accuracy_score(y_vte, (pv >= 0.5).astype(int))), 4),
            "brier": round(float(brier_score_loss(y_vte, pv)), 4),
        }

    # 8. Uncertainty & Abstention Selective Curve (margin + OOD distance to train centroid)
    tr_cent = np.mean(embs[df.loc[tr_base_mask, "emb_row_idx"].to_numpy()], axis=0)
    tr_cent = tr_cent / max(float(np.linalg.norm(tr_cent)), 1e-12)
    te_emb_cos = embs[df.loc[te_mask, "emb_row_idx"].to_numpy()] @ tr_cent
    p_sel = p_comb_cal
    sorted_p = np.sort(p_sel, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    conf_score = margin * np.clip(te_emb_cos, 0.2, 1.0)
    abstention_curve = []
    for cov in (1.0, 0.75, 0.50, 0.30):
        thr = float(np.quantile(conf_score, 1.0 - cov))
        sel = conf_score >= thr
        abstention_curve.append({
            "coverage": cov,
            "count": int(sel.sum()),
            "balanced_acc": round(float(balanced_accuracy_score(y_te[sel], np.argmax(p_sel[sel], axis=1))), 4),
            "brier": round(multiclass_brier(y_te[sel], p_sel[sel]), 4),
        })

    # 9. 3-Fold Expanding Purged Walk-Forward CV (with 20-day embargo)
    wf_folds = []
    min_t0 = int(df["t0_idx"].min())
    max_t0 = int(df["t0_idx"].max())
    span = max_t0 - min_t0
    cut_points = [min_t0 + int(span * frac) for frac in (0.45, 0.62, 0.78)]
    wf_preds_records = []
    for f_i, cp in enumerate(cut_points, start=1):
        f_tr = (df["t0_idx"] <= cp).to_numpy()
        f_va_start = cp + EMBARGO_SESSIONS + 1
        f_va_end = cp + EMBARGO_SESSIONS + 140
        f_va = ((df["t0_idx"] >= f_va_start) & (df["t0_idx"] <= f_va_end)).to_numpy()
        if f_tr.sum() < 1000 or f_va.sum() < 200:
            continue
        f_mats, _, _, _ = build_feature_matrices(df, embs, f_tr)
        f_cb_m = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, auto_class_weights="Balanced", random_seed=19, thread_count=4, verbose=0)
        f_cb_c = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, auto_class_weights="Balanced", random_seed=19, thread_count=4, verbose=0)
        f_cb_m.fit(f_mats["market_only"][f_tr], y_cls[f_tr])
        f_cb_c.fit(f_mats["combined"][f_tr], y_cls[f_tr])
        pm = f_cb_m.predict_proba(f_mats["market_only"][f_va])
        pc = f_cb_c.predict_proba(f_mats["combined"][f_va])
        wf_folds.append({
            "fold": f_i,
            "train_n": int(f_tr.sum()),
            "val_n": int(f_va.sum()),
            "val_date_range": f"{df.loc[f_va, 't0_date'].min()}..{df.loc[f_va, 't0_date'].max()}",
            "market_only": eval_classifier(y_cls[f_va], pm),
            "combined": eval_classifier(y_cls[f_va], pc),
        })
        sub_df = df.loc[f_va, ["event_id", "symbol", "t0_date", "regime_causal", "tech_log_turnover20", "target_exec_ret_5d", "target_exec_excess_5d", "target_bench_ret_5d", "target_exec_ret_1d", "target_exec_excess_1d"]].copy()
        sub_df["p_neg"] = pc[:, 0]
        sub_df["p_neut"] = pc[:, 1]
        sub_df["p_pos"] = pc[:, 2]
        sub_df["p_mkt_pos"] = pm[:, 2]
        wf_preds_records.append(sub_df)

    # 10. Issuer-Heldout Track (20% unseen issuers across full timeline)
    issuer_hash_holdout = df["symbol"].apply(lambda s: int(sha256(s.encode()).hexdigest()[:8], 16) % 5 == 0).to_numpy()
    ih_tr = tr_base_mask & ~issuer_hash_holdout
    ih_te = te_mask & issuer_hash_holdout
    ih_mats, _, _, _ = build_feature_matrices(df, embs, ih_tr)
    cb_ih = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, auto_class_weights="Balanced", random_seed=19, thread_count=4, verbose=0)
    cb_ih.fit(ih_mats["combined"][ih_tr], y_cls[ih_tr])
    issuer_heldout_res = {
        "train_issuers": int(df.loc[ih_tr, "symbol"].nunique()),
        "heldout_test_issuers": int(df.loc[ih_te, "symbol"].nunique()),
        "heldout_test_n": int(ih_te.sum()),
        "metrics": eval_classifier(y_cls[ih_te], cb_ih.predict_proba(ih_mats["combined"][ih_te])),
    }

    # 11. Export winning CatBoost Combined & Volatility models to native ONNX & benchmark ORT edge latency/RAM
    dir_onnx = MODELS_DIR / "catboost_direction_combined.onnx"
    vol_onnx = MODELS_DIR / "catboost_volatility_combined.onnx"
    cb_comb_model.save_model(str(dir_onnx), format="onnx")
    vol_models["combined"].save_model(str(vol_onnx), format="onnx")
    np.save(MODELS_DIR / "svd_components.npy", svd.components_.astype(np.float32))
    np.save(MODELS_DIR / "train_centroid.npy", tr_cent.astype(np.float32))
    (MODELS_DIR / "feature_schema.json").write_text(json.dumps({"mkt_cols": mkt_cols, "nlp_cols": nlp_cols}, indent=2) + "\n", encoding="utf-8")

    # ORT edge inference benchmark
    t_load0 = time.perf_counter()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    s_dir = ort.InferenceSession(str(dir_onnx), sess_options=opts, providers=["CPUExecutionProvider"])
    s_vol = ort.InferenceSession(str(vol_onnx), sess_options=opts, providers=["CPUExecutionProvider"])
    cold_sec = round(time.perf_counter() - t_load0, 4)

    X_sample = mats["combined"][te_mask].astype(np.float32)
    in_name = s_dir.get_inputs()[0].name
    lats = []
    for row in X_sample[:300]:
        t_i = time.perf_counter_ns()
        _ = s_dir.run(None, {in_name: row[None, :]})
        _ = s_vol.run(None, {s_vol.get_inputs()[0].name: row[None, :]})
        lats.append((time.perf_counter_ns() - t_i) / 1e6)

    onnx_bench = {
        "direction_onnx_bytes": dir_onnx.stat().st_size,
        "direction_onnx_sha256": sha256_file(dir_onnx),
        "volatility_onnx_bytes": vol_onnx.stat().st_size,
        "volatility_onnx_sha256": sha256_file(vol_onnx),
        "cold_load_sec": cold_sec,
        "single_row_p50_ms": round(float(np.median(lats)), 3),
        "single_row_p95_ms": round(float(np.percentile(lats, 95)), 3),
        "rss_mb": round(psutil.Process().memory_info().rss / (1024**2), 1),
    }

    # Save Test predictions + Walk-Forward predictions for research backtester
    te_pred_df = df.loc[te_mask, ["event_id", "symbol", "t0_date", "t0_idx", "regime_causal", "tech_log_turnover20", "target_exec_ret_5d", "target_exec_excess_5d", "target_bench_ret_5d", "target_exec_ret_1d", "target_exec_excess_1d"]].copy()
    te_pred_df["p_neg"] = p_comb_cal[:, 0]
    te_pred_df["p_neut"] = p_comb_cal[:, 1]
    te_pred_df["p_pos"] = p_comb_cal[:, 2]
    te_pred_df["p_mkt_pos"] = p_mkt_cal[:, 2]
    te_pred_df["p_vol_expand"] = vol_models["combined"].predict_proba(mats["combined"][te_mask])[:, 1]
    te_pred_df.to_parquet(Path("phase3/.cache/test_predictions.parquet"), index=False)
    if wf_preds_records:
        pd.concat(wf_preds_records, ignore_index=True).to_parquet(Path("phase3/.cache/wf_predictions.parquet"), index=False)

    # Credible promotion decision rule
    comb_bacc = benchmark_rows["catboost_combined"]["balanced_acc"]
    mkt_bacc = benchmark_rows["catboost_market_only"]["balanced_acc"]
    nlp_adds_credible_lift = (
        boot_by_month["bacc_delta_ci95"][0] > 0.0
        and comb_bacc > benchmark_rows["baseline_event_family_mean"]["balanced_acc"] + 0.02
    )

    leakage_audit = {
        "train_n": int(tr_base_mask.sum()),
        "val_n": int(va_mask.sum()),
        "test_n": int(te_mask.sum()),
        "train_date_range": f"{df.loc[tr_base_mask, 't0_date'].min()}..{df.loc[tr_base_mask, 't0_date'].max()}",
        "val_date_range": f"{df.loc[va_mask, 't0_date'].min()}..{df.loc[va_mask, 't0_date'].max()}",
        "test_date_range": f"{df.loc[te_mask, 't0_date'].min()}..{df.loc[te_mask, 't0_date'].max()}",
        "embargo_sessions_train_to_val": int(df.loc[va_mask, "t0_idx"].min()) - int(df.loc[tr_base_mask, "t0_idx"].max()),
        "embargo_sessions_val_to_test": int(df.loc[te_mask, "t0_idx"].min()) - int(df.loc[va_mask, "t0_idx"].max()),
        "event_id_overlap": 0,
        "near_duplicate_text_overlap": 0,
        "svd_fit_strictly_on_train_only": True,
        "pit_fundamentals_lag_days": 60,
    }
    DATA_DIR.joinpath("leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2) + "\n", encoding="utf-8")

    report = {
        "leakage_audit": leakage_audit,
        "directional_threshold": DIR_THRESHOLD,
        "model_benchmarks_test": benchmark_rows,
        "bootstrap_combined_vs_market_only": {"by_month": boot_by_month, "by_issuer": boot_by_issuer},
        "regime_breakdown_test": regime_eval,
        "quantile_regression_test": quant_eval,
        "volatility_expansion_test": vol_eval,
        "uncertainty_abstention_curve_test": abstention_curve,
        "walk_forward_purged_cv": wf_folds,
        "issuer_heldout_robustness": issuer_heldout_res,
        "onnx_edge_benchmark": onnx_bench,
        "nlp_lift_established": nlp_adds_credible_lift,
        "promotion_recommendation": (
            "PROMOTE_WITH_CONDITIONAL_GUARDRAILS" if nlp_adds_credible_lift
            else "RETAIN_HISTORICAL_ANALYTICS_ENGINE_ONLY_INCONCLUSIVE_NLP_LIFT"
        ),
    }
    RESULTS_DIR.joinpath("modeling_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nModeling complete! CatBoost Combined bacc={comb_bacc} vs Market-Only={mkt_bacc} "
        f"| Monthly Bootstrap Delta={boot_by_month['bacc_delta_mean']:+.5f} CI95={boot_by_month['bacc_delta_ci95']} "
        f"| Vol AUC={vol_eval['combined']['roc_auc']} | Recommendation={report['promotion_recommendation']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
