"""Zero-dependency text magnitude & tone signal extractors shared across Phase 3 build and edge runtime."""
from __future__ import annotations

import re
import numpy as np

MONEY_RE = re.compile(r"(?:₹|Rs\.?|INR|USD|\$)\s*\d[\d,]*(?:\.\d+)?\s*(?:crores?|crs?|lakhs?|lacs?|millions?|mn|billions?|bn)?", re.I)
PCT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
NEG_TONE_RE = re.compile(r"\b(?:denies|refutes|baseless|false|cancelled|terminated|penalty|fine|default|downgrade|loss|decline|fell|slump|strike|outage|show\s+cause)\b", re.I)
POS_TONE_RE = re.compile(r"\b(?:approved|completed|awarded|bagged|growth|rose|increased|dividend|buyback|bonus|upgrade|launch|record\s+revenue|profit\s+jumped)\b", re.I)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
