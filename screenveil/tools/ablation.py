"""Measure how much inference work ScreenVeil's change + scroll tracking saves
versus naively analysing every band of every frame (demo video)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from screenveil.conductor import NPUConductor  # noqa: E402
from screenveil.engine import EngineConfig, ScreenVeilEngine  # noqa: E402
from screenveil.state import ScreenState  # noqa: E402
from screenveil.types import Line  # noqa: E402

sys.path.insert(0, str(ROOT / "demo"))
from make_demo import demo_frames  # noqa: E402

FRAMES = demo_frames()


class CountingText:
    """Zero-cost stand-in text engine: we only count how often it is called."""
    name, device = "count", "CPU"

    def __init__(self):
        self.calls = 0

    def devices(self):
        return ("CPU",)

    def set_device(self, d, m):
        return "CPU"

    def read_lines(self, bgr) -> list[Line]:
        self.calls += 1
        return []


def run(scroll: bool) -> tuple[int, int, int]:
    orig = ScreenState._detect_scroll
    if not scroll:
        ScreenState._detect_scroll = staticmethod(lambda a, b, factor=4: 0)
    try:
        text = CountingText()
        eng = ScreenVeilEngine(text, NPUConductor(forced_mode="performance"),
                               config=EngineConfig(detect_faces=False))
        frames = 0
        for f in FRAMES:                      # lossless frames: no codec noise
            eng.process(f)
            frames += 1
        return frames, text.calls, len(eng.state.layout.bands)
    finally:
        ScreenState._detect_scroll = orig


if __name__ == "__main__":
    frames, calls, bands = run(scroll=True)
    _, calls_noscroll, _ = run(scroll=False)
    naive = frames * bands
    r = {"frames": frames, "bands_per_frame": bands, "naive_band_inferences": naive,
         "screenveil_band_inferences": calls, "without_scroll_tracking": calls_noscroll,
         "work_saved_pct": round(100 * (1 - calls / naive), 1),
         "scroll_tracking_saves_pct": round(100 * (1 - calls / calls_noscroll), 1)}
    print(json.dumps(r, indent=1))
    (ROOT / "docs" / "ablation.json").write_text(json.dumps(r, indent=1))
