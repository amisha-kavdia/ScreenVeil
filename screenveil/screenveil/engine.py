"""The ScreenVeil pipeline: capture -> change tracking -> scheduled analysis
on the NPU -> classification -> fail-closed composition."""
from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .classify import ClassifierConfig, SensitiveTextClassifier
from .conductor import NPUConductor
from .redact import compose, full_shield
from .state import BandLayout, ScreenState
from .types import Finding

log = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    band_h: int = 256
    overlap: int = 96
    cell: int = 16
    show_labels: bool = True
    detect_faces: bool = True
    max_errors: int = 3              # consecutive backend errors before full shield
    max_pending_s: float = 3.0       # informational: report if analysis falls this far behind
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)


class ScreenVeilEngine:
    def __init__(self, text_engine, conductor: NPUConductor, face_engine=None,
                 config: EngineConfig | None = None):
        self.cfg = config or EngineConfig()
        self.text = text_engine
        self.faces = face_engine if self.cfg.detect_faces else None
        self.conductor = conductor
        self.classifier = SensitiveTextClassifier(self.cfg.classifier)
        self.state: ScreenState | None = None
        conductor.register("text", text_engine)
        if self.faces is not None:
            conductor.register("faces", self.faces)
        self.errors = 0
        self.kind_counts: collections.Counter = collections.Counter()
        self.bands_analysed = 0
        self.frames = 0
        self._sweep_ptr = 0
        self.last_timing: dict[str, float] = {}

    # ------------------------------------------------------------------ setup
    def calibrate(self, sample: np.ndarray) -> dict:
        """Measure each model on each available device using one band of `sample`."""
        band = sample[: min(self.cfg.band_h, sample.shape[0])]
        out = {"text": self.conductor.calibrate("text", lambda: self.text.read_lines(band))}
        if self.faces is not None:
            out["faces"] = self.conductor.calibrate("faces", lambda: self.faces.detect(band))
        return out

    def _ensure_state(self, frame: np.ndarray) -> ScreenState:
        h, w = frame.shape[:2]
        if self.state is None or (self.state.layout.width, self.state.layout.height) != (w, h):
            self.state = ScreenState(BandLayout(w, h, self.cfg.band_h, self.cfg.overlap, self.cfg.cell))
        return self.state

    # ---------------------------------------------------------------- per frame
    def process(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        st = self._ensure_state(frame)
        st.observe(frame)
        plan = self.conductor.plan(st.oldest_pending_age())
        t_obs = time.perf_counter()

        bands = st.next_bands(plan.band_budget)
        if not bands and plan.do_sweep:              # idle: re-verify one clean band
            bands = [self._sweep_ptr % len(st.layout.bands)]
            self._sweep_ptr += 1
        deadline = t_obs + plan.time_budget_ms / 1000.0
        infer_s = 0.0
        for i, b in enumerate(bands):
            if i > 0 and time.perf_counter() > deadline:
                break                                 # leave the rest pixelated for next frame
            try:
                ti = time.perf_counter()
                findings = self._analyse_band(frame, b, plan.run_faces)
                infer_s += time.perf_counter() - ti
                st.commit_band(b, findings)
                self.bands_analysed += 1
                self.errors = 0
            except Exception:                         # fail closed, keep going
                log.exception("analysis of band %d failed", b)
                self.errors += 1
                st.requeue(b)

        if self.errors >= self.cfg.max_errors:
            out = full_shield(frame)
        else:
            out = compose(frame, st.unverified, st.layout.cell, st.findings, self.cfg.show_labels)
        total = time.perf_counter() - t0
        self.conductor.record_overhead(max(0.0, total - infer_s))
        self.conductor.frame_done()
        self.frames += 1
        self.last_timing = {"total_ms": total * 1e3, "inference_ms": infer_s * 1e3,
                            "bands": len(bands)}
        return out

    def _analyse_band(self, frame: np.ndarray, b: int, run_faces: bool) -> list[Finding]:
        y0, y1 = self.state.layout.bands[b]
        band = frame[y0:y1]
        t = time.perf_counter()
        lines = self.text.read_lines(band)
        self.conductor.record("text", time.perf_counter() - t)
        found = [Finding(f.kind, f.box.shifted(dy=y0), f.confidence, f.preview, f.source)
                 for f in self.classifier.classify_lines(lines)]
        if self.faces is not None and run_faces:
            t = time.perf_counter()
            for box, conf in self.faces.detect(band):
                found.append(Finding("FACE", box.shifted(dy=y0), conf, "", "vision"))
            self.conductor.record("faces", time.perf_counter() - t)
        elif self.faces is not None:
            # face model skipped this tick to save power: carry forward faces
            # already known in this band so they stay covered
            c0, c1 = self.state.layout.cores[b]
            found.extend(f for f in self.state.findings if f.kind == "FACE" and c0 <= f.box.cy < c1)
        for f in found:
            self.kind_counts[f.kind] += 1
        return found

    # ---------------------------------------------------------------- telemetry
    def snapshot(self) -> dict:
        snap = self.conductor.snapshot()
        st = self.state
        snap.update({
            "frames": self.frames,
            "bands_analysed": self.bands_analysed,
            "pending_pct": round(100 * st.pending_fraction(), 1) if st else 100.0,
            "active_redactions": len(st.findings) if st else 0,
            "detections_by_kind": dict(self.kind_counts),
            "last_frame": {k: round(v, 2) for k, v in self.last_timing.items()},
            "healthy": self.errors < self.cfg.max_errors,
            "last_scroll_px": st.last_scroll if st else 0,
        })
        return snap
