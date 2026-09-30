"""Neural OCR for the Snapdragon NPU: CRAFT text detector + CRNN/CTC recogniser.

These are the two halves of EasyOCR, which Qualcomm AI Hub publishes as an
on-device model (`qai_hub_models.models.easyocr`). `tools/export_models.py`
exports them as ONNX for Snapdragon X; this engine runs them through ONNX
Runtime's QNN execution provider.

The NPU wants static shapes, which is why ScreenVeil analyses the screen as
fixed-height horizontal *bands*: every band is letterboxed to the detector's
fixed input size, so one compiled graph serves every frame.

Pre/post-processing (score-map decoding, CTC decoding) is pure numpy/OpenCV
and unit-tested in `tests/test_onnx_ocr_post.py`.
"""
from __future__ import annotations

import math
import string

import cv2
import numpy as np

from ..classify import group_lines
from ..types import Box, Line, Word
from .ort_session import available_devices, make_session

# EasyOCR 'english_g2' character set (index 0 is the CTC blank).
EASYOCR_EN_CHARSET = ("0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
                      "€" + string.ascii_uppercase + string.ascii_lowercase)


# ---------------------------------------------------------------- detection
def letterbox(img: np.ndarray, out_h: int, out_w: int) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio and pad bottom/right. Returns (image, scale)."""
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((out_h, out_w) + img.shape[2:], dtype=img.dtype)
    canvas[:nh, :nw] = resized
    return canvas, scale


def craft_boxes(text_map: np.ndarray, link_map: np.ndarray, text_threshold: float = 0.7,
                link_threshold: float = 0.4, low_text: float = 0.4,
                min_area: int = 10) -> list[tuple[int, int, int, int]]:
    """Decode CRAFT region/affinity score maps into word boxes (map coords).

    Follows the reference CRAFT post-processing: threshold both maps, take
    connected components, keep components whose peak region score passes
    `text_threshold`, and dilate each proportionally to its size.
    """
    h, w = text_map.shape
    text_bin = text_map >= low_text
    link_bin = link_map >= link_threshold
    combined = np.clip(text_bin.astype(np.uint8) + link_bin.astype(np.uint8), 0, 1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=4)
    boxes = []
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area < min_area:
            continue
        comp = labels == k
        if text_map[comp].max() < text_threshold:
            continue
        niter = int(math.sqrt(area * min(bw, bh) / (bw * bh)) * 2)
        x0, y0 = max(0, x - niter), max(0, y - niter)
        x1, y1 = min(w, x + bw + niter + 1), min(h, y + bh + niter + 1)
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


# -------------------------------------------------------------- recognition
def ctc_greedy_decode(logits: np.ndarray, charset: str) -> tuple[str, float]:
    """Greedy CTC decode of a (T, C) score matrix; blank = index 0."""
    if logits.ndim == 3:
        logits = logits[0]
    x = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
    idx = probs.argmax(axis=1)
    out, confs, prev = [], [], -1
    for t, i in enumerate(idx):
        if i != prev and i != 0 and i - 1 < len(charset):
            out.append(charset[i - 1])
            confs.append(float(probs[t, i]))
        prev = i
    conf = float(np.exp(np.mean(np.log(np.maximum(confs, 1e-6))))) if confs else 0.0
    return "".join(out), conf


def prepare_rec_crop(gray: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a word crop to the recogniser's fixed (H, W), normalise to [-1, 1]."""
    h, w = gray.shape[:2]
    nw = min(out_w, max(1, int(round(w * out_h / max(1, h)))))
    resized = cv2.resize(gray, (nw, out_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((out_h, out_w), resized[:, -1:].mean() if nw else 0, dtype=np.float32)
    canvas[:, :nw] = resized
    canvas = (canvas / 255.0 - 0.5) / 0.5
    return canvas[None, None].astype(np.float32)


class OnnxOcrEngine:
    name = "craft+crnn-onnx"

    def __init__(self, detector_path: str, recognizer_path: str,
                 charset: str = EASYOCR_EN_CHARSET, device: str = "NPU",
                 power_mode: str = "balanced", min_conf: float = 0.05):
        self.det_path, self.rec_path = detector_path, recognizer_path
        self.charset = charset
        self.min_conf = min_conf
        self.device = device
        self.set_device(device, power_mode)

    def devices(self) -> tuple[str, ...]:
        return available_devices()

    def set_device(self, device: str, power_mode: str) -> str:
        self.det, used = make_session(self.det_path, device, power_mode)
        self.rec, _ = make_session(self.rec_path, device, power_mode)
        di = self.det.get_inputs()[0]
        self.det_nchw = di.shape[1] == 3
        self.det_h, self.det_w = (di.shape[2], di.shape[3]) if self.det_nchw else (di.shape[1], di.shape[2])
        ri = self.rec.get_inputs()[0]
        self.rec_h, self.rec_w = int(ri.shape[2]), int(ri.shape[3])
        self.device = used
        return used

    def _detect(self, bgr: np.ndarray) -> list[Box]:
        img, scale = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.det_h, self.det_w)
        x = img.astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406], np.float32) * 255
        std = np.array([0.229, 0.224, 0.225], np.float32) * 255
        x = (x - mean) / std
        x = x.transpose(2, 0, 1)[None] if self.det_nchw else x[None]
        y = self.det.run(None, {self.det.get_inputs()[0].name: x})[0][0]   # (H/2, W/2, 2)
        if y.shape[0] == 2:                                                # channels-first variant
            y = y.transpose(1, 2, 0)
        ratio = (self.det_h / y.shape[0]) / scale
        boxes = []
        for bx, by, bw, bh in craft_boxes(y[..., 0], y[..., 1]):
            boxes.append(Box(int(bx * ratio), int(by * ratio),
                             max(1, int(bw * ratio)), max(1, int(bh * ratio))))
        return boxes

    def read_lines(self, bgr: np.ndarray) -> list[Line]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        words: list[Word] = []
        name = self.rec.get_inputs()[0].name
        for b in self._detect(bgr):
            b = b.clipped(gray.shape[1], gray.shape[0])
            if b is None or b.h < 4:
                continue
            crop = gray[b.y:b.y2, b.x:b.x2]
            text, conf = ctc_greedy_decode(
                self.rec.run(None, {name: prepare_rec_crop(crop, self.rec_h, self.rec_w)})[0],
                self.charset)
            if text.strip() and conf >= self.min_conf:
                # a box may hold several words; split so the classifier sees tokens
                parts = text.split()
                n = max(1, len(text))
                pos = 0
                for p in parts:
                    start = text.index(p, pos)
                    pos = start + len(p)
                    x0 = b.x + int(b.w * start / n)
                    words.append(Word(p, Box(x0, b.y, max(1, int(b.w * len(p) / n)), b.h), conf))
        return group_lines(words)
