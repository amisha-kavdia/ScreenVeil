import unittest

import numpy as np

from screenveil.conductor import NPUConductor
from screenveil.engine import EngineConfig, ScreenVeilEngine
from screenveil.state import BandLayout, ScreenState
from screenveil.types import Box, Finding, Line, Word


class Layout(unittest.TestCase):
    def test_cores_tile_screen_and_keep_margin(self):
        for h in (720, 1080, 1200, 300, 257):
            L = BandLayout(1920, h)
            self.assertEqual(L.cores[0][0], 0)
            self.assertEqual(L.cores[-1][1], h)
            for (a, b), (c, d) in zip(L.cores, L.cores[1:]):
                self.assertEqual(b, c)
            for i, ((y0, y1), (c0, c1)) in enumerate(zip(L.bands, L.cores)):
                # a line of height <= overlap/2 touching the core is inside the band
                if i > 0:
                    self.assertGreaterEqual(c0 - y0, L.overlap // 2)
                if i < len(L.bands) - 1:
                    self.assertGreaterEqual(y1 - c1, L.overlap // 2)


def page(h=2000, w=640, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 240, np.uint8)
    for y in range(20, h - 20, 40):                       # "text lines"
        x = int(rng.integers(10, 200))
        img[y:y + 14, x:x + int(rng.integers(100, 400))] = rng.integers(0, 80)
    return img


class ScrollAndVerify(unittest.TestCase):
    def test_everything_unverified_at_start(self):
        st = ScreenState(BandLayout(640, 480))
        st.observe(page()[:480])
        self.assertEqual(st.pending_fraction(), 1.0)

    def test_commit_verifies_core(self):
        st = ScreenState(BandLayout(640, 480))
        st.observe(page()[:480])
        for b in range(len(st.layout.bands)):
            st.commit_band(b, [])
        self.assertEqual(st.pending_fraction(), 0.0)
        self.assertEqual(st.queue, {})

    def test_change_requeues_band(self):
        st = ScreenState(BandLayout(640, 480))
        f = page()[:480]
        st.observe(f)
        for b in range(len(st.layout.bands)):
            st.commit_band(b, [])
        g = f.copy()
        g[300:320, 100:200] = 0
        st.observe(g)
        self.assertGreater(st.pending_fraction(), 0)
        self.assertIn(st.layout.band_of_y(310), st.queue)

    def test_scroll_moves_redactions_and_only_exposes_new_strip(self):
        p = page()
        st = ScreenState(BandLayout(640, 480))
        st.observe(p[0:480])
        for b in range(len(st.layout.bands)):
            st.commit_band(b, [Finding("PAN", Box(50, 200, 100, 16), 1.0)] if b == st.layout.band_of_y(208) else [])
        st.observe(p[64:544])                            # scroll down by 64 px
        self.assertEqual(st.last_scroll, -64)
        self.assertEqual(st.findings[0].box.y, 136)      # redaction moved with content
        self.assertLess(st.pending_fraction(), 0.3)      # most content still verified

    def test_sticky_findings_survive_one_miss(self):
        st = ScreenState(BandLayout(640, 480))
        st.observe(page()[:480])
        b = st.layout.band_of_y(208)
        st.commit_band(b, [Finding("PAN", Box(50, 200, 100, 16), 1.0)])
        st.commit_band(b, [])                            # OCR flickers
        self.assertEqual(len(st.findings), 1)


class FakeText:
    name, device = "fake", "CPU"

    def __init__(self, fail=False):
        self.fail = fail

    def devices(self):
        return ("CPU",)

    def set_device(self, d, m):
        return "CPU"

    def read_lines(self, bgr):
        if self.fail:
            raise RuntimeError("NPU crashed")
        return [Line([Word("PAN", Box(10, 20, 30, 16)), Word("ABCPK1234Z", Box(50, 20, 100, 16))])]


class EngineFailClosed(unittest.TestCase):
    def test_redacts_after_analysis(self):
        eng = ScreenVeilEngine(FakeText(), NPUConductor(forced_mode="performance"),
                               config=EngineConfig(detect_faces=False))
        f = page()[:480]
        for _ in range(5):
            out = eng.process(f)
        self.assertEqual(eng.state.pending_fraction(), 0.0)
        self.assertTrue(any(x.kind == "PAN" for x in eng.state.findings))
        self.assertFalse(np.array_equal(out[20:36, 50:150], f[20:36, 50:150]))

    def test_backend_failure_shields_whole_screen(self):
        eng = ScreenVeilEngine(FakeText(fail=True), NPUConductor(forced_mode="performance"),
                               config=EngineConfig(detect_faces=False))
        f = page()[:480]
        for _ in range(4):
            out = eng.process(f)
        self.assertFalse(eng.snapshot()["healthy"])
        self.assertLess(out.mean(), f.mean() * 0.5)      # dimmed full shield

    def test_unanalysed_content_never_passes_through(self):
        eng = ScreenVeilEngine(FakeText(), NPUConductor(forced_mode="critical"),
                               config=EngineConfig(detect_faces=False))
        f = page()[:480]
        out = eng.process(f)                             # budget: only some bands done
        L = eng.state.layout
        for r in range(L.rows):
            if eng.state.unverified[r].all():
                y = r * L.cell
                self.assertFalse(np.array_equal(out[y:y + L.cell], f[y:y + L.cell]))


if __name__ == "__main__":
    unittest.main()
