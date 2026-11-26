"""Deterministic financial numerical reasoning engine with provenance, unit normalization, and safety guards."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

# Base unit multipliers:
# - INR family -> INR crores (1 crore = 1e7 INR, 1 lakh = 1e5 INR = 0.01 crore)
# - USD/EUR/GBP family -> base currency units (million = 1e6, billion = 1e9, thousand = 1e3)
SCALE_INR_CRORE = {
    "": 1e-7,
    "rupee": 1e-7,
    "rupees": 1e-7,
    "inr": 1e-7,
    "thousand": 1e-4,
    "k": 1e-4,
    "lakh": 0.01,
    "lakhs": 0.01,
    "lac": 0.01,
    "lacs": 0.01,
    "million": 0.1,
    "mn": 0.1,
    "crore": 1.0,
    "crores": 1.0,
    "cr": 1.0,
    "crs": 1.0,
    "billion": 100.0,
    "bn": 100.0,
}

SCALE_GENERIC = {
    "": 1.0,
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "percent": 1.0,
    "%": 1.0,
    "bps": 0.01,  # 100 bps = 1 percentage point
}

AMOUNT_PARSE_RE = re.compile(
    r"(?P<sign>[-−])?\s*(?P<curr>₹|Rs\.?|INR|USD|\$|EUR|€|GBP|£)?\s*"
    r"(?P<num>\d+(?:,\d+)*(?:\.\d+)?)\s*"
    r"(?P<scale>crores?|crs?|lakhs?|lacs?|millions?|mn|billions?|bn|thousands?|bps|%)?",
    re.I,
)


@dataclass(frozen=True)
class SourceQuantity:
    raw_text: str
    value: float
    currency: str
    scale: str
    normalized_value: float
    normalized_unit: str
    period: str = ""
    page: int = 1
    start: int = 0
    end: int = 0


def parse_quantity(
    text: str,
    *,
    default_currency: str = "",
    default_scale: str = "",
    period: str = "",
    page: int = 1,
    start: int = 0,
) -> SourceQuantity | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    paren_neg = cleaned.startswith("(") and cleaned.endswith(")")
    probe = cleaned[1:-1].strip() if paren_neg else cleaned
    m = AMOUNT_PARSE_RE.search(probe)
    if not m or not m.group("num"):
        return None
    val = float(m.group("num").replace(",", ""))
    if m.group("sign") or paren_neg:
        val = -val

    raw_curr = (m.group("curr") or default_currency or "").upper().rstrip(".")
    curr_map = {"₹": "INR", "RS": "INR", "INR": "INR", "$": "USD", "USD": "USD", "€": "EUR", "EUR": "EUR", "£": "GBP", "GBP": "GBP"}
    currency = curr_map.get(raw_curr, raw_curr)

    scale = (m.group("scale") or default_scale or "").lower()
    if scale == "%" or "%" in probe:
        return SourceQuantity(cleaned, val, "PCT", "%", val, "%", period, page, start, start + len(cleaned))
    if scale == "bps":
        return SourceQuantity(cleaned, val, "PCT", "bps", val * 0.01, "%", period, page, start, start + len(cleaned))

    if currency == "INR" or scale in ("crore", "crores", "cr", "crs", "lakh", "lakhs", "lac", "lacs"):
        mult = SCALE_INR_CRORE.get(scale, 1e-7)
        return SourceQuantity(cleaned, val, "INR", scale or "unit", val * mult, "INR_crore", period, page, start, start + len(cleaned))

    mult = SCALE_GENERIC.get(scale, 1.0)
    norm_unit = currency or "UNIT"
    return SourceQuantity(cleaned, val, currency or "NONE", scale or "unit", val * mult, norm_unit, period, page, start, start + len(cleaned))


def _unanswerable(reason: str, formula: str, sources: list[SourceQuantity]) -> dict[str, Any]:
    return {
        "status": "unanswerable",
        "reason": reason,
        "value": None,
        "unit": None,
        "formula": formula,
        "source_values": [asdict(s) for s in sources],
        "evidence": [{"page": s.page, "start": s.start, "end": s.end, "text": s.raw_text} for s in sources],
    }


def compute_metric(
    operation: str,
    a: SourceQuantity | None,
    b: SourceQuantity | None = None,
    *,
    require_same_currency: bool = True,
    check_period_compat: bool = False,
) -> dict[str, Any]:
    """Safely compute financial metrics with full provenance and guardrails."""
    if a is None:
        return _unanswerable("missing_primary_operand", operation, [])
    sources = [a] if b is None else [a, b]

    if b is not None and require_same_currency:
        if a.currency not in ("NONE", "") and b.currency not in ("NONE", "") and a.currency != b.currency:
            return _unanswerable("currency_mismatch_no_fx_rate", f"{operation}({a.raw_text}, {b.raw_text})", sources)
        if a.normalized_unit != b.normalized_unit and not (a.currency == "NONE" or b.currency == "NONE"):
            return _unanswerable("incompatible_units", f"{operation}({a.raw_text}, {b.raw_text})", sources)

    if check_period_compat and a.period and b is not None and b.period:
        a_is_q = a.period.upper().startswith("Q")
        b_is_q = b.period.upper().startswith("Q")
        if a_is_q != b_is_q:
            return _unanswerable("inconsistent_periods", f"{operation}({a.period} vs {b.period})", sources)

    if operation in ("yoy_growth", "qoq_growth", "pct_change", "debt_change_pct", "earnings_growth"):
        if b is None:
            return _unanswerable("missing_baseline_denominator", operation, sources)
        if abs(b.normalized_value) < 1e-12:
            return _unanswerable("zero_denominator", f"({a.normalized_value} - {b.normalized_value}) / {b.normalized_value}", sources)
        if b.normalized_value < 0:
            diff = a.normalized_value - b.normalized_value
            return {
                "status": "swing_or_negative_base",
                "reason": "negative_denominator_pct_undefined",
                "value": round(diff, 4),
                "unit": a.normalized_unit,
                "formula": f"({a.normalized_value:.6g} - ({b.normalized_value:.6g})) -> absolute_change ({a.normalized_unit}); pct undefined for negative base",
                "source_values": [asdict(s) for s in sources],
                "evidence": [{"page": s.page, "start": s.start, "end": s.end, "text": s.raw_text} for s in sources],
            }
        pct = ((a.normalized_value - b.normalized_value) / b.normalized_value) * 100.0
        return {
            "status": "ok",
            "reason": None,
            "value": round(pct, 4),
            "unit": "%",
            "formula": f"(({a.normalized_value:.6g} - {b.normalized_value:.6g}) / {b.normalized_value:.6g}) * 100",
            "source_values": [asdict(s) for s in sources],
            "evidence": [{"page": s.page, "start": s.start, "end": s.end, "text": s.raw_text} for s in sources],
        }

    if operation in ("profit_margin", "operating_margin", "ratio"):
        if b is None:
            return _unanswerable("missing_denominator", operation, sources)
        if abs(b.normalized_value) < 1e-12 or b.normalized_value < 0:
            return _unanswerable("invalid_or_zero_denominator", f"{a.normalized_value} / {b.normalized_value}", sources)
        val = (a.normalized_value / b.normalized_value) * (100.0 if "margin" in operation else 1.0)
        return {
            "status": "ok",
            "reason": None,
            "value": round(val, 4),
            "unit": "%" if "margin" in operation else "ratio",
            "formula": f"({a.normalized_value:.6g} / {b.normalized_value:.6g})" + (" * 100" if "margin" in operation else ""),
            "source_values": [asdict(s) for s in sources],
            "evidence": [{"page": s.page, "start": s.start, "end": s.end, "text": s.raw_text} for s in sources],
        }

    if operation in ("difference", "debt_change", "revenue_diff", "profit_diff"):
        if b is None:
            return _unanswerable("missing_second_operand", operation, sources)
        diff = a.normalized_value - b.normalized_value
        return {
            "status": "ok",
            "reason": None,
            "value": round(diff, 4),
            "unit": a.normalized_unit,
            "formula": f"{a.normalized_value:.6g} - {b.normalized_value:.6g}",
            "source_values": [asdict(s) for s in sources],
            "evidence": [{"page": s.page, "start": s.start, "end": s.end, "text": s.raw_text} for s in sources],
        }

    if operation in ("pp_difference", "bps_difference"):
        if b is None:
            return _unanswerable("missing_second_operand", operation, sources)
        if a.currency != "PCT" or b.currency != "PCT":
            return _unanswerable("expected_percentage_operands", operation, sources)
        pp = a.normalized_value - b.normalized_value
        val = pp * 100.0 if operation == "bps_difference" else pp
        return {
            "status": "ok",
            "reason": None,
            "value": round(val, 4),
            "unit": "bps" if operation == "bps_difference" else "pp",
            "formula": f"({a.normalized_value:.6g} - {b.normalized_value:.6g})" + (" * 100" if operation == "bps_difference" else ""),
            "source_values": [asdict(s) for s in sources],
            "evidence": [{"page": s.page, "start": s.start, "end": s.end, "text": s.raw_text} for s in sources],
        }

    return _unanswerable(f"unsupported_operation:{operation}", operation, sources)


def execute_finqa_program(program_tokens: list[str], table: list[list[str]] | None = None) -> dict[str, Any]:
    """Deterministic FinQA DSL executor (`add(a, b)`, `subtract(a, b)`, `multiply(a, b)`, `divide(a, b)`, `table_*`)."""
    # FinQA program string is serialized as steps like: ['subtract(100, 80)', 'divide(#0, 80)', 'EOF']
    results: list[float] = []
    steps_log: list[str] = []

    def resolve(arg: str) -> float:
        arg = arg.strip()
        if arg.startswith("#"):
            idx = int(arg[1:])
            return results[idx]
        if arg.startswith("const_"):
            val = arg[len("const_"):]
            return float(val.replace("m1", "-1"))
        if arg.endswith("%"):
            return float(arg[:-1].replace(",", "")) / 100.0
        return float(arg.replace(",", "").replace("$", ""))

    def table_row_vals(row_name: str) -> list[float]:
        if not table:
            raise ValueError("missing_table")
        target = row_name.strip().lower()
        for row in table:
            if row and row[0].strip().lower() == target:
                vals = []
                for cell in row[1:]:
                    q = parse_quantity(cell)
                    if q is not None and math.isfinite(q.value):
                        vals.append(q.value)
                if vals:
                    return vals
        raise ValueError(f"table_row_not_found:{row_name}")

    try:
        for tok in program_tokens:
            t = tok.strip()
            if not t or t == "EOF":
                break
            m = re.match(r"([a-z_]+)\((.+)\)$", t)
            if not m:
                return {"status": "error", "reason": f"invalid_step:{t}", "value": None, "formula": " -> ".join(steps_log)}
            op, raw_args = m.group(1), m.group(2)
            parts = [p.strip() for p in raw_args.split(",")]
            if op == "add":
                val = resolve(parts[0]) + resolve(parts[1])
            elif op == "subtract":
                val = resolve(parts[0]) - resolve(parts[1])
            elif op == "multiply":
                val = resolve(parts[0]) * resolve(parts[1])
            elif op == "divide":
                den = resolve(parts[1])
                if abs(den) < 1e-12:
                    return {"status": "unanswerable", "reason": "zero_denominator", "value": None, "formula": " -> ".join(steps_log)}
                val = resolve(parts[0]) / den
            elif op == "exp":
                val = resolve(parts[0]) ** resolve(parts[1])
            elif op == "greater":
                val = 1.0 if resolve(parts[0]) > resolve(parts[1]) else 0.0
            elif op.startswith("table_"):
                vals = table_row_vals(parts[0])
                if op == "table_sum":
                    val = sum(vals)
                elif op == "table_average":
                    val = sum(vals) / len(vals)
                elif op == "table_max":
                    val = max(vals)
                elif op == "table_min":
                    val = min(vals)
                else:
                    return {"status": "error", "reason": f"unknown_table_op:{op}", "value": None, "formula": " -> ".join(steps_log)}
            else:
                return {"status": "error", "reason": f"unknown_op:{op}", "value": None, "formula": " -> ".join(steps_log)}
            results.append(val)
            steps_log.append(f"#{len(results) - 1} = {t} -> {val:.6g}")
    except (ValueError, IndexError, ZeroDivisionError) as exc:
        return {"status": "unanswerable", "reason": str(exc), "value": None, "formula": " -> ".join(steps_log)}

    if not results:
        return {"status": "unanswerable", "reason": "empty_program", "value": None, "formula": ""}
    return {"status": "ok", "reason": None, "value": round(results[-1], 6), "formula": " ; ".join(steps_log)}
