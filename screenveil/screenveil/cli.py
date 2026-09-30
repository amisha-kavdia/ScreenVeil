"""Command line: `python -m screenveil <command>`."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from . import __version__
from .backends.ort_session import available_devices
from .classify import ClassifierConfig
from .conductor import NPUConductor
from .engine import EngineConfig, ScreenVeilEngine

MODELS = Path(os.environ.get("SCREENVEIL_MODELS", Path(__file__).resolve().parent.parent / "models"))
DET = MODELS / "easyocr_detector.onnx"
REC = MODELS / "easyocr_recognizer.onnx"


def build_engine(args) -> ScreenVeilEngine:
    choice = args.engine
    if choice == "auto":
        choice = "onnx" if DET.exists() and REC.exists() else "tesseract"
    if choice == "onnx":
        from .backends.onnx_ocr import OnnxOcrEngine
        text = OnnxOcrEngine(str(args.det or DET), str(args.rec or REC), device=available_devices()[0])
    else:
        from .backends.tesseract_engine import TesseractTextEngine
        text = TesseractTextEngine()
    faces = None
    if not args.no_faces:
        from .backends.faces import HaarFaceEngine
        faces = HaarFaceEngine()
    power_w = json.loads(Path(args.power_profile).read_text()) if args.power_profile else None
    conductor = NPUConductor(forced_mode=args.mode, power_w=power_w)
    terms = [t.strip() for t in (args.terms or "").split(",") if t.strip()]
    cfg = EngineConfig(detect_faces=not args.no_faces, show_labels=not args.no_labels,
                       classifier=ClassifierConfig(custom_terms=terms))
    return ScreenVeilEngine(text, conductor, faces, cfg)


def make_source(args):
    from .io import ImageSource, ScreenSource, VideoSource
    if args.video:
        return VideoSource(args.video)
    if args.images:
        return ImageSource(args.images, repeat=args.repeat)
    return ScreenSource(args.monitor, args.window)


def cmd_run(args) -> int:
    from .io import PreviewSink, RateLimiter, VideoSink, VirtualCamSink
    eng = build_engine(args)
    src = make_source(args)
    first = src.read()
    if first is None:
        print("no frames from source", file=sys.stderr)
        return 1
    print("calibrating models on available devices:", available_devices())
    print(json.dumps(eng.calibrate(first), indent=1))
    sinks = []
    if not args.headless:
        sinks.append(PreviewSink())
    if args.virtualcam:
        sinks.append(VirtualCamSink(first.shape[1], first.shape[0]))
    if args.record:
        fps = getattr(src, "fps", 20)
        sinks.append(VideoSink(args.record, fps, (first.shape[1], first.shape[0])))
    if args.dashboard:
        from .dashboard import serve
        serve(eng.snapshot, args.dashboard)
        print(f"dashboard: http://127.0.0.1:{args.dashboard}")
    limiter = RateLimiter()
    frame = first
    try:
        while frame is not None:
            out = eng.process(frame)
            if not all(s.write(out) for s in sinks):
                break
            if not args.video:                     # screen / stills: pace like live capture
                limiter.wait(eng.conductor.target_fps())
            frame = src.read()
    except KeyboardInterrupt:
        pass
    finally:
        for s in sinks:
            s.close()
    print(json.dumps(eng.snapshot(), indent=1))
    return 0


def cmd_bench(args) -> int:
    """Process a video/image set headless and report latency statistics."""
    eng = build_engine(args)
    src = make_source(args)
    frame = src.read()
    calib = eng.calibrate(frame)
    totals, infers = [], []
    t0 = time.perf_counter()
    n = 0
    while frame is not None:
        eng.process(frame)
        totals.append(eng.last_timing["total_ms"])
        infers.append(eng.last_timing["inference_ms"])
        n += 1
        frame = src.read()
    wall = time.perf_counter() - t0
    totals.sort()
    report = {
        "frames": n, "wall_s": round(wall, 2), "throughput_fps": round(n / wall, 1),
        "frame_ms_p50": round(totals[len(totals) // 2], 1),
        "frame_ms_p95": round(totals[int(len(totals) * 0.95) - 1], 1),
        "calibration_s": calib, "devices": list(available_devices()),
        "snapshot": eng.snapshot(),
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


def cmd_devices(args) -> int:
    import onnxruntime as ort
    print("ScreenVeil", __version__)
    print("onnxruntime", ort.__version__, ort.get_available_providers())
    print("usable devices (best first):", ", ".join(available_devices()))
    print("NPU OCR models present:", DET.exists() and REC.exists(), f"({MODELS})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="screenveil", description="On-device privacy shield for screen sharing")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "bench"):
        s = sub.add_parser(name)
        s.add_argument("--monitor", type=int, default=1)
        s.add_argument("--window", help="capture only the window whose title contains this (Windows)")
        s.add_argument("--video", help="process a video file instead of the screen")
        s.add_argument("--images", nargs="+", help="process still images instead of the screen")
        s.add_argument("--repeat", type=int, default=1, help="repeat each image N frames")
        s.add_argument("--engine", choices=["auto", "onnx", "tesseract"], default="auto")
        s.add_argument("--det"), s.add_argument("--rec")
        s.add_argument("--mode", choices=["performance", "balanced", "saver", "critical"],
                       help="force a power mode (default: follow the battery)")
        s.add_argument("--terms", help="comma-separated confidential words, e.g. client or project names")
        s.add_argument("--power-profile", help="JSON of measured watts per device (tools/calibrate_power.py)")
        s.add_argument("--no-faces", action="store_true")
        s.add_argument("--no-labels", action="store_true")
        s.add_argument("--out", help="bench: write JSON report here")
        if name == "run":
            s.add_argument("--headless", action="store_true")
            s.add_argument("--virtualcam", action="store_true")
            s.add_argument("--record", help="also save the protected output as .mp4")
            s.add_argument("--dashboard", type=int, nargs="?", const=8765, default=8765)
    sub.add_parser("devices")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    return {"run": cmd_run, "bench": cmd_bench, "devices": cmd_devices}[args.cmd](args)
