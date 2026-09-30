"""Run ScreenVeil over the demo video and write:
  demo/assets/demo_protected.mp4    - what meeting attendees would see
  demo/assets/demo_side_by_side.mp4 - original | protected
  demo/assets/after_*.png           - protected stills of each screen
  demo/assets/demo_report.json      - timing + detections

Frames are processed exactly as in live mode (same per-frame budget, same
Conductor decisions); the only difference is the source is a file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from screenveil.backends.faces import HaarFaceEngine  # noqa: E402
from screenveil.backends.tesseract_engine import TesseractTextEngine  # noqa: E402
from screenveil.classify import ClassifierConfig  # noqa: E402
from screenveil.conductor import NPUConductor  # noqa: E402
from screenveil.engine import EngineConfig, ScreenVeilEngine  # noqa: E402

A = ROOT / "demo" / "assets"


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (20, 20, 24), -1)
    cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def main():
    cfg = EngineConfig(classifier=ClassifierConfig(custom_terms=["Project Falcon"]))
    eng = ScreenVeilEngine(TesseractTextEngine(), NPUConductor(forced_mode="performance"),
                           HaarFaceEngine(), cfg)
    sys.path.insert(0, str(ROOT / "demo"))
    from make_demo import demo_frames
    frames = demo_frames()          # lossless, like a real screen capture
    fps = 15
    frame = frames[0]
    eng.calibrate(frame)
    h, w = frame.shape[:2]
    prot = cv2.VideoWriter(str(A / "demo_protected.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    sbs = cv2.VideoWriter(str(A / "demo_side_by_side.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (w, h // 2))
    timings, pending, idx = [], [], 0
    stills = {19: "bank", 50: "scroll", 64: "bank_scrolled", 101: "chat", 126: "code"}
    for frame in frames:
        out = eng.process(frame)
        timings.append(eng.last_timing["total_ms"])
        pending.append(eng.state.pending_fraction())
        prot.write(out)
        pair = np.hstack([label(frame, "What you see"), label(out, "What the meeting sees")])
        sbs.write(cv2.resize(pair, (w, h // 2), interpolation=cv2.INTER_AREA))
        if idx in stills:
            cv2.imwrite(str(A / f"after_{stills[idx]}.png"), out)
            cv2.imwrite(str(A / f"before_{stills[idx]}.png"), frame)
        idx += 1
    prot.release()
    sbs.release()
    t = sorted(timings)
    report = {
        "frames": idx,
        "frame_ms_p50": round(t[len(t) // 2], 1),
        "frame_ms_p95": round(t[int(len(t) * 0.95) - 1], 1),
        "mean_unverified_pct": round(100 * float(np.mean(pending)), 1),
        "frames_fully_verified_pct": round(100 * float(np.mean([p == 0 for p in pending])), 1),
        "snapshot": eng.snapshot(),
    }
    (A / "demo_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
