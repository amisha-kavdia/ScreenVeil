"""Backend interfaces. Every backend works on a BGR numpy image (one band of
the screen) and returns results in that band's coordinates."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..types import Box, Line


class TextEngine(Protocol):
    name: str
    device: str                      # device currently in use: NPU / GPU / CPU

    def devices(self) -> tuple[str, ...]:
        """Devices this engine can run on."""

    def set_device(self, device: str, power_mode: str) -> str:
        """Switch device / power mode; returns the device actually used."""

    def read_lines(self, bgr: np.ndarray) -> list[Line]:
        """Detect + recognise text lines in the image."""


class ObjectEngine(Protocol):
    name: str
    device: str
    kind: str                        # finding kind, e.g. FACE

    def devices(self) -> tuple[str, ...]: ...

    def set_device(self, device: str, power_mode: str) -> str: ...

    def detect(self, bgr: np.ndarray) -> list[tuple[Box, float]]: ...
