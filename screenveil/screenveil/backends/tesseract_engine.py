"""CPU fallback text engine using Tesseract (runs anywhere).

Used for development on non-Snapdragon machines and as the "CPU-only
baseline" the NPU Conductor compares against.
"""
from __future__ import annotations

import os

# Tesseract's OpenMP threading costs ~4x on small images; one thread per call is fastest.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import cv2  # noqa: E402
import numpy as np

from ..classify import group_lines
from ..types import Box, Line, Word


class TesseractTextEngine:
    name = "tesseract-ocr"

    def __init__(self, upscale: float = 1.0, min_conf: float = 0.0, lang: str = "eng"):
        import pytesseract
        cmd = os.environ.get("TESSERACT_CMD")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        elif os.name == "nt" and os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(
                "Tesseract OCR is not installed or not on PATH. Install it "
                "(Windows: https://github.com/UB-Mannheim/tesseract/wiki) or set TESSERACT_CMD "
                "to tesseract.exe, or add the NPU models to models/ (see README).") from e
        self.upscale = upscale
        self.min_conf = min_conf
        self.lang = lang
        self.device = "CPU"

    def devices(self) -> tuple[str, ...]:
        return ("CPU",)

    def set_device(self, device: str, power_mode: str) -> str:
        return "CPU"

    def read_lines(self, bgr: np.ndarray) -> list[Line]:
        import pytesseract

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 110:          # dark themes: OCR prefers dark text on light
            gray = 255 - gray
        s = self.upscale
        if s != 1.0:
            gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(gray, lang=self.lang, config="--oem 1 --psm 11",
                                         output_type=pytesseract.Output.DICT)
        words: list[Word] = []
        for i, text in enumerate(data["text"]):
            text = (text or "").strip()
            conf = float(data["conf"][i])
            # NB: random strings (keys, tokens) get near-zero confidence because
            # they are not dictionary words - exactly the text we must not drop.
            if not text or conf < self.min_conf:
                continue
            box = Box(int(data["left"][i] / s), int(data["top"][i] / s),
                      max(1, int(round(data["width"][i] / s))),
                      max(1, int(round(data["height"][i] / s))))
            words.append(Word(text, box, conf / 100.0))
        return group_lines(words)
