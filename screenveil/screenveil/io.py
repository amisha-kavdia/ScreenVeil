"""Frame sources and sinks."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------ sources
class ScreenSource:
    """Captures a monitor, or the rectangle of one window (Windows), via mss."""

    def __init__(self, monitor: int = 1, window_title: str | None = None, max_width: int | None = 1920):
        import mss
        self.sct = mss.mss()
        self.monitor = monitor
        self.window_title = window_title
        self.max_width = max_width

    def _region(self) -> dict:
        if self.window_title and sys.platform == "win32":
            rect = find_window_rect(self.window_title)
            if rect:
                l, t, r, b = rect
                return {"left": l, "top": t, "width": r - l, "height": b - t}
        return self.sct.monitors[self.monitor]

    def read(self) -> np.ndarray | None:
        img = np.asarray(self.sct.grab(self._region()))[:, :, :3]
        if self.max_width and img.shape[1] > self.max_width:
            s = self.max_width / img.shape[1]
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img)


def find_window_rect(title_part: str):  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if title_part.lower() in buf.value.lower() and "screenveil" not in buf.value.lower():
                found.append(hwnd)
                return False
        return True

    user32.EnumWindows(cb, 0)
    if not found:
        return None
    r = wintypes.RECT()
    user32.GetWindowRect(found[0], ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


class VideoSource:
    def __init__(self, path: str, loop: bool = False):
        self.cap = cv2.VideoCapture(path)
        self.loop = loop
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None


class ImageSource:
    def __init__(self, paths: list[str], repeat: int = 1):
        self.frames = [cv2.imread(str(p)) for p in paths for _ in range(repeat)]
        self.i = 0

    def read(self):
        if self.i >= len(self.frames):
            return None
        self.i += 1
        return self.frames[self.i - 1]


# -------------------------------------------------------------------- sinks
class PreviewSink:
    """A window showing the protected feed. Share *this window* in Teams /
    Zoom / Meet instead of your screen. Keep it on a second monitor, or use
    --window to capture a single app so the preview is not captured itself."""

    title = "ScreenVeil - share this window"

    def __init__(self, max_width: int = 1600):
        self.max_width = max_width
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)

    def write(self, frame: np.ndarray) -> bool:
        if frame.shape[1] > self.max_width:
            s = self.max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imshow(self.title, frame)
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)

    def close(self):
        cv2.destroyWindow(self.title)


class VirtualCamSink:
    """Optional: publish as a virtual camera (needs `pyvirtualcam` and a
    virtual-camera driver such as the OBS Virtual Camera)."""

    def __init__(self, width: int, height: int, fps: float = 20):
        import pyvirtualcam
        self.cam = pyvirtualcam.Camera(width=width, height=height, fps=fps)

    def write(self, frame: np.ndarray) -> bool:
        self.cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return True

    def close(self):
        self.cam.close()


class VideoSink:
    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> bool:
        self.w.write(frame)
        return True

    def close(self):
        self.w.release()


class RateLimiter:
    def __init__(self):
        self.t = time.perf_counter()

    def wait(self, fps: float) -> None:
        period = 1.0 / max(1.0, fps)
        now = time.perf_counter()
        sleep = self.t + period - now
        if sleep > 0:
            time.sleep(sleep)
        self.t = max(now, self.t + period)
