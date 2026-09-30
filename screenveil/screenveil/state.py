"""Screen state: bands, change detection, scroll tracking, verification grid.

Key idea — **no unverified pixel ever leaves the machine.**

The screen is split into full-width horizontal *bands* (text runs
horizontally, so a band never cuts a line of text in half) that overlap by
`overlap` px. Each band *owns* a "core" strip; cores tile the screen exactly.
A text line of height <= overlap/2 that touches a core lies entirely inside
that core's band, so analysing the band is enough to verify the core.

A fine grid of cells tracks which pixels changed since they were last
verified. Changed cells are *unverified*: the compositor pixelates them until
their band has been analysed. That makes ScreenVeil fail-closed — if the NPU
is busy, throttled or crashes, the viewer sees mosaic, never a leak.

Scrolling (the most common "everything changed" event) is detected with
phase correlation; verified content and its redactions move with the scroll,
so only the newly revealed strip needs re-analysis.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .types import Box, Finding


@dataclass
class BandLayout:
    width: int
    height: int
    band_h: int = 256
    overlap: int = 96
    cell: int = 16

    def __post_init__(self):
        bh = min(self.band_h, self.height)
        stride = max(1, bh - self.overlap)
        starts, y = [], 0
        while True:
            starts.append(y)
            if y + bh >= self.height:
                break
            y = min(y + stride, self.height - bh)
        self.bands = [(s, s + bh) for s in starts]
        half = self.overlap // 2
        bounds = [0] + [s + half for s in starts[1:]] + [self.height]
        self.cores = [(bounds[i], bounds[i + 1]) for i in range(len(starts))]
        self.rows = -(-self.height // self.cell)
        self.cols = -(-self.width // self.cell)
        centers = (np.arange(self.rows) + 0.5) * self.cell
        self.row_band = np.searchsorted([c[1] for c in self.cores], centers, side="right")
        self.row_band = np.minimum(self.row_band, len(self.bands) - 1)

    def band_of_y(self, y: float) -> int:
        for i, (a, b) in enumerate(self.cores):
            if a <= y < b:
                return i
        return len(self.cores) - 1


@dataclass
class _Tracked:
    finding: Finding
    last_seen: int                      # frame index when last (re)detected


@dataclass
class ScreenState:
    layout: BandLayout
    diff_threshold: int = 14            # grey-level change that counts as "changed"
    sticky_frames: int = 6              # keep a vanished finding this many frames (OCR flicker)
    frame_idx: int = 0
    prev: np.ndarray | None = None
    unverified: np.ndarray = field(init=False)
    queue: dict[int, float] = field(default_factory=dict)   # band -> time first dirtied
    tracked: list[_Tracked] = field(default_factory=list)
    last_scroll: int = 0

    def __post_init__(self):
        L = self.layout
        self.unverified = np.ones((L.rows, L.cols), dtype=bool)
        now = time.perf_counter()
        self.queue = {i: now for i in range(len(L.bands))}

    # ---------------------------------------------------------------- input
    def observe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Register a new frame; returns the boolean grid of changed cells."""
        self.frame_idx += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        L = self.layout
        if self.prev is None or self.prev.shape != gray.shape:
            self.prev = gray
            changed = np.ones((L.rows, L.cols), dtype=bool)
            self._mark(changed)
            return changed

        dy = self._detect_scroll(self.prev, gray)
        self.last_scroll = dy
        ref = self.prev.astype(np.int16)
        exposed = np.zeros(gray.shape, dtype=bool)
        if dy:
            ref = np.roll(ref, dy, axis=0)
            if dy > 0:
                exposed[:dy] = True
            else:
                exposed[dy:] = True
            self._apply_scroll(dy)
        diff = (np.abs(gray.astype(np.int16) - ref) >= self.diff_threshold) | exposed
        changed = self._cell_any(diff)
        self._mark(changed)
        self.prev = gray
        return changed

    def _cell_any(self, mask: np.ndarray) -> np.ndarray:
        L = self.layout
        pad = np.zeros((L.rows * L.cell, L.cols * L.cell), dtype=bool)
        pad[:mask.shape[0], :mask.shape[1]] = mask
        return pad.reshape(L.rows, L.cell, L.cols, L.cell).any(axis=(1, 3))

    def _mark(self, changed: np.ndarray) -> None:
        if not changed.any():
            return
        self.unverified |= changed
        now = time.perf_counter()
        for b in np.unique(self.layout.row_band[changed.any(axis=1)]):
            self.queue.setdefault(int(b), now)

    @staticmethod
    def _detect_scroll(prev: np.ndarray, cur: np.ndarray, factor: int = 4) -> int:
        """Vertical scroll offset in px (content moved down = positive), else 0."""
        a = cv2.resize(prev, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)
        b = cv2.resize(cur, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)
        if np.abs(a.astype(np.int16) - b).mean() < 0.5:
            return 0
        (dx, dy), resp = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
        if resp < 0.15 or abs(dx) > 0.6 or abs(dy) < 0.75:
            return 0
        dyi = int(round(dy * factor))
        h = cur.shape[0]
        if abs(dyi) >= h // 2:
            return 0
        # verify: the shifted previous frame must match the overlap well
        # (also guards against sign conventions of phaseCorrelate)
        def err(d: int) -> float:
            if d > 0:
                return float(np.abs(cur[d:].astype(np.int16) - prev[:h - d]).mean())
            return float(np.abs(cur[:h + d].astype(np.int16) - prev[-d:]).mean())

        # the estimate came from a 1/4-scale image: refine to the exact pixel
        # at full resolution, since a 1 px error would dirty every text edge
        cands = {c for base in (dyi, -dyi) for c in range(base - factor, base + factor + 1)
                 if c != 0 and abs(c) < h // 2}
        best = min(cands, key=err, default=0)
        return best if best and err(best) < 3.0 else 0

    def _apply_scroll(self, dy: int) -> None:
        """Move verification state and redactions along with scrolled content."""
        c = self.layout.cell
        lo, hi = dy // c, -(-dy // c)          # floor / ceil in cells (conservative OR)
        g = self.unverified
        shifted = np.roll(g, lo, axis=0) | np.roll(g, hi, axis=0)
        if dy > 0:
            shifted[:hi] = True
        elif dy < 0:
            shifted[lo:] = True
        self.unverified = shifted
        for t in self.tracked:
            f = t.finding
            t.finding = Finding(f.kind, f.box.shifted(dy=dy), f.confidence, f.preview, f.source)

    # ------------------------------------------------------------- analysis
    def next_bands(self, n: int) -> list[int]:
        """Oldest-dirty-first bands to analyse this tick."""
        return [b for b, _ in sorted(self.queue.items(), key=lambda kv: kv[1])[:max(0, n)]]

    def commit_band(self, band: int, findings: list[Finding]) -> None:
        """Record analysis of `band` (findings already in frame coordinates)."""
        L = self.layout
        c0, c1 = L.cores[band]
        mine = [f for f in findings if c0 <= f.box.cy < c1]
        keep: list[_Tracked] = []
        for t in self.tracked:
            if not (c0 <= t.finding.box.cy < c1):
                keep.append(t)
                continue
            # previously found here: keep briefly if not re-detected (anti-flicker)
            if not any(_iou(t.finding.box, f.box) > 0.3 and f.kind == t.finding.kind for f in mine) \
                    and self.frame_idx - t.last_seen <= self.sticky_frames:
                keep.append(t)
        keep.extend(_Tracked(f, self.frame_idx) for f in mine)
        self.tracked = keep
        self.unverified[L.row_band == band] = False
        self.queue.pop(band, None)

    def requeue(self, band: int) -> None:
        self.queue.setdefault(band, time.perf_counter())

    @property
    def findings(self) -> list[Finding]:
        return [t.finding for t in self.tracked]

    def oldest_pending_age(self) -> float:
        if not self.queue:
            return 0.0
        return time.perf_counter() - min(self.queue.values())

    def pending_fraction(self) -> float:
        return float(self.unverified.mean())


def _iou(a: Box, b: Box) -> float:
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0
