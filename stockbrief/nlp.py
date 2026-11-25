"""Event classification and page-aware evidence extraction."""
from dataclasses import asdict, dataclass
import re
from time import perf_counter_ns

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, make_pipeline

from .corpus import CALIBRATION, LABELS, TRAIN
from .fixtures import Document

MODEL = "Word + character TF-IDF / logistic regression"
MAX_CHUNKS = 240
CHUNK_SIZE = 780
PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z₹0-9])|\n\s*\n")
ABBREVIATION = re.compile(r"\b(?:Mr|Ms|Dr|Ltd|Inc|No|Rs|vs)\.$", re.I)
FIGURES = re.compile(
    r"(?:(?:₹|\$|€|£|\b(?:INR|USD|EUR|GBP|Rs\.?))\s*\d+(?:[,.]\d+)*"
    r"(?:\s*(?:crores?|lakhs?|million|billion)\b)?(?:\s+per share\b)?"
    r"|\b\d+(?:[,.]\d+)*\s*%(?:\s*(?:to|–|-)\s*\d+(?:[,.]\d+)*\s*%)?"
    r"|\b\d+\s*:\s*\d+\b)",
    re.I,
)
KEYWORDS = {
    "earnings": ("financial results", "earnings", "net profit", "guidance", "operating margin", "quarterly revenue"),
    "corporate_action": ("dividend", "buyback", "bonus share", "stock split", "record date", "rights issue"),
    "deal": ("acquire", "acquisition", "takeover", "merger", "demerger", "share purchase agreement"),
    "governance": ("director", "chief financial officer", "resignation", "auditor", "chief executive"),
    "regulatory": ("penalty", "regulator", "show cause", "tribunal", "enforcement", "tax demand", "fine"),
    "operations": ("production", "contract", "plant", "outage", "operations", "purchase order"),
    "routine": ("trading window", "annual general meeting", "conference call", "annual report", "voting", "webcast"),
}
STAGES = {
    "deal": (
        ("Denied", r"\b(?:denied|no agreement (?:was |has been )?signed|did not (?:acquire|complete)|no acquisition (?:has |was )?(?:been )?(?:made|completed))\b"),
        ("Terminated", r"\b(?:terminat\w+|abandoned|withdrew|cancelled|canceled|called off)\b"),
        (
            "Completed",
            r"\b(?:(?:acquisition|transaction|deal|merger|demerger) "
            r"(?:has (?:been )?|was |is )?(?:now )?(?:closed|completed|consummated)|"
            r"(?:closed|completed|consummated|confirmed closing of) (?:the |its |this )?"
            r"(?:previously announced )?(?:acquisition|transaction|deal|merger|demerger))\b",
        ),
        ("Agreement signed", r"\b(?:definitive agreement|signed (?:an? )?agreement|share purchase agreement was executed)\b"),
        ("Subject to approvals", r"\b(?:subject to|remains conditional|pending)\b.{0,150}\b(?:approval|clearance|conditions)"),
        ("Board approved", r"\b(?:board approved|directors approved|board endorsed|approved (?:the |a )?(?:composite )?scheme)\b"),
        ("Proposed", r"\b(?:proposed|proposal|considering|plans to)\b"),
    ),
    "regulatory": (
        ("Penalty imposed", r"\b(?:imposed|levied)\b.{0,80}\b(?:fine|penalty|sanction)\b"),
        ("Under appeal", r"\b(?:appeal|stayed|challenge)\b"),
        ("Notice received", r"\b(?:show cause|notice to show cause)\b"),
    ),
    "corporate_action": (
        ("Recommended", r"\brecommend\w+\b"),
        ("Declared", r"\b(?:declared|fixed|will receive)\b"),
        ("Approved", r"\bapprov\w+\b"),
    ),
    "earnings": (("Reported", r"\b(?:report\w+|publish\w+|approved|results show)\b"),),
    "pricing": (
        ("Denied", r"\b(?:denied|no price (?:increase|hike)|will not (?:raise|increase|cut) prices)\b"),
        ("Announced", r"\b(?:announc\w+|price revision|will (?:raise|increase|cut|reduce) prices|"
                      r"introduced|launched|is increasing|is reducing)\b"),
    ),
    "governance": (
        ("Resignation", r"\b(?:resign\w+|steps? down|step[p]?ed down|cease to hold office)\b"),
        ("Appointment", r"\b(?:appoint\w+|named|nominated)\b"),
    ),
    "operations": (
        ("Disruption reported", r"\b(?:suspended|disruption|halted|outage|fire)\b"),
        ("Operations resumed", r"\b(?:resumed|restarted|commenced|begun)\b"),
        ("Contract awarded", r"\b(?:awarded|won|letter of award)\b"),
    ),
}
COMPLETION_BLOCKERS = re.compile(
    r"\b(?:not yet|not been|not complete[ds]?|no closing|subject to|conditional|"
    r"upon completion|pending|expected to|once|after completion of)\b", re.I
)


@dataclass(frozen=True)
class Chunk:
    page: int
    start: int
    end: int
    text: str


def chunks_for(pages: tuple[str, ...], pattern=PATTERN) -> list[Chunk]:
    found = []
    for page_number, page in enumerate(pages, start=1):
        start = 0
        boundaries = []
        for boundary in pattern.finditer(page):
            if not ABBREVIATION.search(page[max(0, boundary.start() - 8):boundary.start()]):
                boundaries.append((boundary.start(), boundary.end()))
        boundaries.append((len(page), len(page)))
        for stop, next_start in boundaries:
            while start < stop and len(found) < MAX_CHUNKS:
                cut = min(start + CHUNK_SIZE, stop)
                if cut < stop:
                    nearby_space = page.rfind(" ", start + CHUNK_SIZE // 2, cut)
                    cut = nearby_space if nearby_space > start else cut
                content = page[start:cut]
                stripped = content.strip()
                if len(stripped) >= 12:
                    left = start + len(content) - len(content.lstrip())
                    found.append(Chunk(page_number, left, left + len(stripped), stripped))
                start = cut
            start = next_start
        if len(found) >= MAX_CHUNKS:
            break
    return found


def baseline(text: str) -> str:
    normalized = text.lower()
    hits = {
        label: sum(bool(re.search(r"\b" + re.escape(term) + r"\b", normalized)) for term in terms)
        for label, terms in KEYWORDS.items()
    }
    maximum = max(hits.values())
    winners = [label for label, count in hits.items() if count == maximum]
    return winners[0] if maximum and len(winners) == 1 else "review"


def stage_for(label: str, candidates: list[Chunk]) -> tuple[str, dict | None]:
    for stage, pattern in STAGES.get(label, ()):
        for chunk in candidates:
            normalized = " ".join(chunk.text.split())
            if stage == "Completed" and COMPLETION_BLOCKERS.search(normalized):
                continue
            if stage == "Disruption reported" and re.search(r"\b(?:restarted|resumed)\b", normalized, re.I):
                continue
            if re.search(pattern, normalized, re.I):
                return stage, asdict(chunk)
    return "Unclear", None


class EventModel:
    name = MODEL

    def __init__(self):
        samples = [(text, label) for label, texts in TRAIN.items() for text in texts]
        texts, targets = zip(*samples)
        self.pipeline = make_pipeline(
            FeatureUnion(
                [
                    ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                    ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
                ],
                transformer_weights={"words": 1.0, "characters": 1.2},
            ),
            LogisticRegression(C=6, max_iter=600, random_state=19),
        ).fit(texts, targets)
        self.classes = self.pipeline.classes_.tolist()
        self.floor, self.margin = self.calibrate()

    def scores(self, chunks: list[Chunk]) -> tuple[np.ndarray, np.ndarray]:
        texts = [chunk.text for chunk in chunks]
        sentence_scores = self.pipeline.predict_proba(texts)
        combined = self.pipeline.predict_proba([" ".join(texts)])[0]
        # Retain the strongest sentence for each class, while document context limits single-word effects.
        scores = 0.65 * combined + 0.35 * sentence_scores.max(axis=0)
        return scores / scores.sum(), sentence_scores

    @staticmethod
    def choose(scores: np.ndarray, classes: list[str], floor: float, margin: float) -> tuple[str, str, float]:
        order = np.argsort(scores)
        best = int(order[-1])
        score = float(scores[best])
        winner = classes[best]
        if score < floor or score - float(scores[order[-2]]) < margin:
            return "review", winner, score
        return winner, winner, score

    def calibrate(self) -> tuple[float, float]:
        fixtures = [(text, label) for label, texts in CALIBRATION.items() for text in texts]
        labels = [label for _, label in fixtures]
        values = [self.scores(chunks_for((text,)))[0] for text, _ in fixtures]
        grid = [(floor, margin) for floor in (0.19, 0.22, 0.25, 0.28, 0.31) for margin in (0.02, 0.04, 0.06, 0.08)]
        return max(
            grid,
            key=lambda pair: (
                f1_score(labels, [self.choose(value, self.classes, *pair)[0] for value in values], average="macro"),
                pair[0] + pair[1],
            ),
        )

    def analyze(self, document: Document) -> dict:
        started = perf_counter_ns()
        chunks = chunks_for(document.pages)
        if not chunks:
            raise ValueError("Provide at least 12 characters of readable disclosure text.")
        scores, sentence_scores = self.scores(chunks)
        label, candidate, score = self.choose(scores, self.classes, self.floor, self.margin)
        class_index = self.classes.index(candidate)
        rankings = np.argsort(-sentence_scores[:, class_index], kind="stable")
        ranked = [chunks[int(i)] for i in rankings]
        related = [chunks[int(i)] for i in rankings if int(np.argmax(sentence_scores[int(i)])) == class_index]
        context = related if related else ranked[:1]
        # Nearby continuation clauses can carry status or terms even when they are not event-bearing on their own.
        adjacent = {
            neighbor for rank in rankings[:2] for neighbor in (int(rank) - 1, int(rank) + 1)
            if 0 <= neighbor < len(chunks)
        }
        context = list(dict.fromkeys(context[:5] + [chunks[i] for i in sorted(adjacent)]))
        context = sorted(context, key=lambda item: (item.page, item.start))
        stage, stage_evidence = stage_for(label, context)
        evidence = [asdict(chunk) for chunk in ranked[:2]]
        if stage_evidence and stage_evidence not in evidence:
            evidence.append(stage_evidence)
        figures = []
        for chunk in context if label != "review" else ():
            for match in FIGURES.finditer(chunk.text):
                if len(figures) < 8:
                    figures.append({"value": match.group(), "page": chunk.page, "start": chunk.start + match.start()})
        return {
            "category": label,
            "category_name": LABELS.get(label, "Needs review"),
            "candidate": candidate,
            "candidate_name": LABELS[candidate],
            "score": round(score, 4),
            "scores": {key: round(float(value), 4) for key, value in zip(self.classes, scores)},
            "stage": stage,
            "stage_evidence": stage_evidence,
            "evidence": evidence,
            "figures": figures,
            "chunk_count": len(chunks),
            "chunk_limit_reached": len(chunks) == MAX_CHUNKS,
            "model": MODEL,
            "inference_ms": round((perf_counter_ns() - started) / 1e6, 3),
        }
