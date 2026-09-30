"""Sensitive-text classifier.

Turns OCR lines into `Finding`s. Design goals:

* Precision matters as much as recall: blurring every number on screen makes a
  presentation useless, so identifiers with a checksum (Aadhaar -> Verhoeff,
  payment cards -> Luhn) are only flagged when the checksum passes.
* OCR is noisy: digit-like tokens are normalised (O->0, l/I->1, S->5, B->8)
  before numeric patterns are applied. Normalisation is 1:1 per character so
  match spans map straight back to word boxes.
* Context matters: an OTP is just "482913" unless the line says OTP / code.
* Everything runs locally; findings only ever carry a masked preview.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .types import Box, Finding, Line, Word, mask_text

# --------------------------------------------------------------------------
# checksums

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(digits: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    return str(_VERHOEFF_INV[c])


def luhn_valid(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


# --------------------------------------------------------------------------
# OCR normalisation

_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1",
                            "|": "1", "S": "5", "B": "8", "Z": "2"})


def _normalise_token(tok: str) -> str:
    """Fix common OCR confusions inside tokens that are mostly digits."""
    core = [c for c in tok if c.isalnum()]
    digits = sum(c.isdigit() for c in core)
    if len(core) >= 4 and digits / len(core) >= 0.6:
        return tok.translate(_DIGIT_FIX)
    # e.g. "4lll": at least one digit and every other char is a digit look-alike
    if len(core) >= 2 and digits and all(c.isdigit() or c in "OoDlI|SBZ" for c in core):
        return tok.translate(_DIGIT_FIX)
    return tok


_CONFUSABLE = set("OoDlI|SBZ")


def _normalise_line(tokens: list[str]) -> list[str]:
    """Normalise digit-like tokens, plus tokens made *only* of digit look-alikes
    ("llll", "OO") when they sit next to a numeric token - OCR often misreads a
    whole 4-digit group of a card or Aadhaar number."""
    out = [_normalise_token(t) for t in tokens]
    numeric = [sum(c.isdigit() for c in t) >= max(1, 0.6 * len(t)) for t in out]
    for i, t in enumerate(tokens):
        if not numeric[i] and t and set(t) <= _CONFUSABLE and len(t) <= 6:
            if (i > 0 and numeric[i - 1]) or (i + 1 < len(out) and numeric[i + 1]):
                out[i] = t.translate(_DIGIT_FIX)
    return out


def _digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


# --------------------------------------------------------------------------
# rules

@dataclass
class Rule:
    kind: str
    pattern: re.Pattern
    confidence: float = 0.9
    validate: Callable[[str], bool] | None = None
    group: int = 0                     # which regex group is the secret (-1: first non-empty)
    context: re.Pattern | None = None  # line must also match this
    numeric: bool = False              # match on the OCR-normalised line


def _aadhaar_ok(s: str) -> bool:
    d = _digits(s)
    return len(d) == 12 and d[0] not in "01" and verhoeff_valid(d)


def _card_ok(s: str) -> bool:
    d = _digits(s)
    return 13 <= len(d) <= 19 and luhn_valid(d) and len(set(d)) > 1


def _secret_ok(s: str) -> bool:
    return len(s) >= 6 and not s.lower() in {"********", "hidden", "none", "null"}


CTX_ACCOUNT = re.compile(r"\b(a/?c|acct|account|khata)\b", re.I)
CTX_OTP = re.compile(r"\b(otp|one[- ]time|verification code|passcode|login code|security code)\b", re.I)
CTX_CVV = re.compile(r"\b(cvv|cvc|card verification)\b", re.I)

DEFAULT_RULES: list[Rule] = [
    # --- Indian identity & banking ---------------------------------------
    Rule("AADHAAR", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)"), 0.97,
         _aadhaar_ok, numeric=True),
    Rule("PAN", re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b"), 0.95),
    Rule("CARD", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), 0.97, _card_ok, numeric=True),
    Rule("IFSC", re.compile(r"\b[A-Z]{4}[0O]{1,2}[A-Z0-9]{6}\b"), 0.8),
    Rule("BANK_ACCOUNT", re.compile(r"(?<!\d)\d(?:[ -]?\d){8,17}(?!\d)"), 0.85,
         context=CTX_ACCOUNT, numeric=True),
    Rule("UPI_ID", re.compile(r"\b[\w.\-]{2,64}@(?:ok[a-z]+|ybl|ibl|axl|apl|upi|paytm|ptyes|"
                              r"ptsbi|pthdfc|ptaxis|jio|fbl|kotak|icici|sbi|hdfcbank|axisbank)\b",
                              re.I), 0.9),
    Rule("PHONE", re.compile(r"(?<![\w\d])(?:\+91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)"), 0.75,
         numeric=True),
    Rule("OTP", re.compile(r"(?<!\d)\d{4,8}(?!\d)"), 0.9, context=CTX_OTP, numeric=True),
    Rule("CVV", re.compile(r"(?<!\d)\d{3,4}(?!\d)"), 0.9, context=CTX_CVV, numeric=True),
    # --- developer secrets ----------------------------------------------
    Rule("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0.99),
    Rule("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), 0.99),
    Rule("GOOGLE_KEY", re.compile(r"\bAIza[0-9A-Za-z\-_]{30,}\b"), 0.97),
    Rule("SLACK_TOKEN", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"), 0.99),
    Rule("STRIPE_KEY", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b"), 0.99),
    Rule("API_KEY", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), 0.95),
    Rule("JWT", re.compile(r"\beyJ[\w\-]{8,}\.eyJ[\w\-]{8,}\.[\w\-]{8,}"), 0.97),
    Rule("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), 0.99),
    Rule("CONN_STRING", re.compile(r"\b\w+:\s?//[^\s:/]+:([^\s@]{3,})@[\w.\-]+"), 0.95, group=1),
    # "password: hunter2", GITHUB_TOKEN = "ghp_...", api_key='...'. The value
    # runs to the closing quote (or end of line), so a token that OCR split
    # into two words is still covered in full.
    Rule("CREDENTIAL", re.compile(
        r"[\w\-]*(?:pass(?:word|wd)?|pwd|secret|api[_ \-]?key|access[_ \-]?key(?:[_ ]?id)?|"
        r"token|auth|private[_ ]?key)[\w\-]*\s*[:=]\s*"
        r"(?:\"([^\"]{4,})\"?|'([^']{4,})'?|([^\s\"']{4,}))", re.I), 0.9, _secret_ok, group=-1),
    Rule("EMAIL", re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b"), 0.8),
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9+/_\-]{24,}")


@dataclass
class ClassifierConfig:
    rules: list[Rule] = field(default_factory=lambda: list(DEFAULT_RULES))
    custom_terms: list[str] = field(default_factory=list)   # e.g. codenames, client names
    enable_entropy: bool = True
    entropy_threshold: float = 4.0
    disabled_kinds: set[str] = field(default_factory=set)
    box_margin: int = 4


class SensitiveTextClassifier:
    def __init__(self, config: ClassifierConfig | None = None):
        self.cfg = config or ClassifierConfig()
        self._custom = [re.compile(r"\b" + re.escape(t) + r"\b", re.I)
                        for t in self.cfg.custom_terms if t.strip()]

    # -- public -------------------------------------------------------------
    def classify_lines(self, lines: Iterable[Line]) -> list[Finding]:
        lines = list(lines)
        out: list[Finding] = []
        for line in lines:
            out.extend(self.classify_line(line, self._context(line, lines)))
        return out

    @staticmethod
    def _context(line: Line, lines: list[Line]) -> str:
        """Text a human would read as this line's label: same visual row
        (label: value laid out in columns) or the line right above it
        (label stacked over a form field)."""
        b = line.box
        parts = []
        for other in lines:
            if other is line or not other.words:
                continue
            o = other.box
            same_row = abs(o.cy - b.cy) <= 0.6 * max(o.h, b.h) and o.x2 <= b.x
            above = 0 <= b.y - o.y2 <= 1.5 * b.h and o.x < b.x2 and b.x < o.x2
            if same_row or above:
                parts.append(" ".join(w.text for w in other.words))
        return " ".join(parts)

    def classify_line(self, line: Line, context: str = "") -> list[Finding]:
        if not line.words:
            return []
        raw, spans = self._join(line.words)
        norm = " ".join(_normalise_line([w.text for w in line.words]))
        assert len(norm) == len(raw)
        findings: list[Finding] = []
        taken: list[tuple[int, int]] = []

        def add(kind: str, start: int, end: int, conf: float, text: str) -> None:
            if kind in self.cfg.disabled_kinds or end <= start:
                return
            for a, b in taken:            # most specific rule wins (rules are ordered)
                if start < b and a < end:
                    return
            box = self._span_box(line.words, spans, start, end)
            if box is None:
                return
            taken.append((start, end))
            findings.append(Finding(kind, box.grown(self.cfg.box_margin), conf, mask_text(text)))

        for rule in self.cfg.rules:
            text = norm if rule.numeric else raw
            if rule.context is not None and not rule.context.search(raw + " " + context):
                continue
            for m in rule.pattern.finditer(text):
                g = rule.group
                if g == -1:
                    g = next((i for i in range(1, (m.lastindex or 0) + 1) if m.group(i)), 0)
                val = m.group(g)
                if rule.validate is not None and not rule.validate(val):
                    continue
                add(rule.kind, m.start(g), m.end(g), rule.confidence, val)

        for rx in self._custom:
            for m in rx.finditer(raw):
                add("CUSTOM_TERM", m.start(), m.end(), 0.95, m.group(0))

        if self.cfg.enable_entropy:
            for m in _TOKEN_RE.finditer(raw):
                tok = m.group(0)
                classes = sum([any(c.islower() for c in tok), any(c.isupper() for c in tok),
                               any(c.isdigit() for c in tok)])
                if classes >= 3 and shannon_entropy(tok) >= self.cfg.entropy_threshold:
                    add("HIGH_ENTROPY_SECRET", m.start(), m.end(), 0.7, tok)
        return findings

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _join(words: list[Word]) -> tuple[str, list[tuple[int, int]]]:
        spans, pos, parts = [], 0, []
        for w in words:
            spans.append((pos, pos + len(w.text)))
            parts.append(w.text)
            pos += len(w.text) + 1
        return " ".join(parts), spans

    @staticmethod
    def _span_box(words: list[Word], spans, start: int, end: int) -> Box | None:
        """Box covering the part of the line between char offsets [start, end).

        Partial words are handled by proportional horizontal slicing, so
        'Password:hunter2' only blurs the value, not the label.
        """
        box = None
        for w, (a, b) in zip(words, spans):
            lo, hi = max(a, start), min(b, end)
            if lo >= hi:
                continue
            n = max(1, b - a)
            cw = w.box.w / n                   # average glyph width in this word
            # proportional slicing assumes equal glyph widths; pad 2 glyphs on
            # any cut edge so variable-width fonts can never leak characters
            pad = max(2 * cw, 0.15 * w.box.w)   # OCR boxes are not evenly spaced
            pad_l = pad if lo > a else 0
            pad_r = pad if hi < b else 0
            x0 = max(w.box.x, w.box.x + int(cw * (lo - a) - pad_l))
            x1 = min(w.box.x2, w.box.x + int(math.ceil(cw * (hi - a) + pad_r)))
            part = Box(x0, w.box.y, max(1, x1 - x0), w.box.h)
            box = part if box is None else box.union(part)
        return box


def group_lines(words: list[Word], y_tol: float = 0.5, gap_factor: float = 2.5) -> list[Line]:
    """Group loose word boxes (e.g. from a neural detector) into text lines.

    Words join a line when their vertical centres are within `y_tol` of the
    line height and the horizontal gap is small relative to text height.
    """
    words = sorted(words, key=lambda w: w.box.x)   # left-to-right, lines grow naturally
    lines: list[Line] = []
    for w in words:
        best = None
        for ln in reversed(lines):
            last = ln.words[-1]
            h = max(last.box.h, w.box.h)
            if abs(last.box.cy - w.box.cy) <= y_tol * h and \
                    -h <= w.box.x - last.box.x2 <= gap_factor * h:
                best = ln
                break
        if best is None:
            lines.append(Line([w]))
        else:
            best.words.append(w)
    for ln in lines:
        ln.words.sort(key=lambda w: w.box.x)
    return lines
