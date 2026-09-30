"""Face detection (faces in photos, webcam previews, ID cards on screen)."""
from __future__ import annotations

import cv2
import numpy as np

from ..types import Box


class HaarFaceEngine:
    """Tiny classical detector that ships with OpenCV: CPU only, ~1-3 ms per band."""
    name = "haar-face"
    kind = "FACE"

    def __init__(self, min_size: int = 28):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.min_size = min_size
        self.device = "CPU"

    def devices(self) -> tuple[str, ...]:
        return ("CPU",)

    def set_device(self, device: str, power_mode: str) -> str:
        return "CPU"

    def detect(self, bgr: np.ndarray) -> list[tuple[Box, float]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found = self.cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6,
                                              minSize=(self.min_size, self.min_size))
        return [(Box(int(x), int(y), int(w), int(h)).grown(int(0.15 * w)), 0.8)
                for (x, y, w, h) in found]
