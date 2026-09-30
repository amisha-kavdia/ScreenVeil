"""Shared lightweight data types."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates (x, y = top-left)."""
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def union(self, other: "Box") -> "Box":
        x, y = min(self.x, other.x), min(self.y, other.y)
        return Box(x, y, max(self.x2, other.x2) - x, max(self.y2, other.y2) - y)

    def shifted(self, dx: int = 0, dy: int = 0) -> "Box":
        return Box(self.x + dx, self.y + dy, self.w, self.h)

    def grown(self, m: int) -> "Box":
        return Box(self.x - m, self.y - m, self.w + 2 * m, self.h + 2 * m)

    def clipped(self, width: int, height: int) -> "Box | None":
        x, y = max(0, self.x), max(0, self.y)
        x2, y2 = min(width, self.x2), min(height, self.y2)
        if x2 <= x or y2 <= y:
            return None
        return Box(x, y, x2 - x, y2 - y)


@dataclass
class Word:
    text: str
    box: Box
    conf: float = 1.0


@dataclass
class Line:
    words: list[Word] = field(default_factory=list)

    @property
    def box(self) -> Box:
        b = self.words[0].box
        for w in self.words[1:]:
            b = b.union(w.box)
        return b


@dataclass
class Finding:
    """Something on screen that must not leave the machine."""
    kind: str            # e.g. AADHAAR, CARD, AWS_KEY, FACE
    box: Box
    confidence: float
    preview: str = ""    # already-masked text, safe to log
    source: str = "text"


def mask_text(s: str, keep: int = 2) -> str:
    """Mask a sensitive string for logs: 'AKIAIOSF...' -> 'AK******'."""
    s = s.strip()
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "*" * min(len(s) - keep, 10)
