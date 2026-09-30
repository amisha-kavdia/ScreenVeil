"""Pre/post-processing of the NPU OCR models (no model files needed)."""
import unittest

import numpy as np

from screenveil.backends.onnx_ocr import (EASYOCR_EN_CHARSET, craft_boxes, ctc_greedy_decode,
                                          letterbox, prepare_rec_crop)


class Post(unittest.TestCase):
    def test_craft_two_words(self):
        text = np.zeros((100, 200), np.float32)
        link = np.zeros_like(text)
        text[40:50, 20:60] = 0.9
        text[40:50, 120:180] = 0.9
        boxes = craft_boxes(text, link)
        self.assertEqual(len(boxes), 2)
        link[42:48, 60:120] = 0.8              # affinity joins them into one word box
        self.assertEqual(len(craft_boxes(text, link)), 1)

    def test_craft_ignores_weak_regions(self):
        text = np.zeros((50, 50), np.float32)
        text[10:20, 10:40] = 0.5               # above low_text but peak < text_threshold
        self.assertEqual(craft_boxes(text, np.zeros_like(text)), [])

    def test_ctc_decode(self):
        cs = EASYOCR_EN_CHARSET
        seq = [cs.index(c) + 1 for c in "AKIA"]
        path = [seq[0], seq[0], 0, seq[1], seq[2], 0, seq[3], 0]   # repeats + blanks
        logits = np.full((len(path), len(cs) + 1), -10.0, np.float32)
        for t, k in enumerate(path):
            logits[t, k] = 10.0
        text, conf = ctc_greedy_decode(logits, cs)
        self.assertEqual(text, "AKIA")
        self.assertGreater(conf, 0.99)

    def test_shapes(self):
        img, s = letterbox(np.zeros((256, 1920, 3), np.uint8), 608, 800)
        self.assertEqual(img.shape, (608, 800, 3))
        self.assertAlmostEqual(s, 800 / 1920)
        x = prepare_rec_crop(np.full((20, 90), 128, np.uint8), 64, 1000)
        self.assertEqual(x.shape, (1, 1, 64, 1000))
        self.assertLessEqual(float(np.abs(x).max()), 1.0)


if __name__ == "__main__":
    unittest.main()
