"""Compile and profile ScreenVeil's models on a real Snapdragon X Elite device
hosted by Qualcomm AI Hub (no Snapdragon laptop needed for this step).

Setup (once):  pip install qai-hub
               qai-hub configure --api_token <token from aihub.qualcomm.com>
Run:           python tools/aihub_profile.py [--device "Snapdragon X Elite CRD"]

Writes docs/aihub_profile.json with per-model on-device latency and which
compute unit (NPU/GPU/CPU) each layer ran on. Put those numbers in the pitch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"text_detector": ROOT / "models" / "easyocr_detector.onnx",
          "text_recognizer": ROOT / "models" / "easyocr_recognizer.onnx"}


def main():
    import qai_hub as hub

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Snapdragon X Elite CRD")
    ap.add_argument("--runtime", default="onnx", help="onnx (ORT+QNN) or qnn_context_binary")
    args = ap.parse_args()
    device = hub.Device(args.device)
    report = {"device": args.device, "runtime": args.runtime, "models": {}}
    for name, path in MODELS.items():
        if not path.exists():
            raise SystemExit(f"missing {path}: run tools/export_models.py first")
        print(f"[{name}] compiling for {args.device} ...")
        cj = hub.submit_compile_job(model=str(path), device=device,
                                    options=f"--target_runtime {args.runtime}")
        target = cj.get_target_model()
        print(f"[{name}] profiling ...")
        pj = hub.submit_profile_job(model=target, device=device)
        prof = pj.download_profile()
        summary = prof.get("execution_summary", {})
        us = summary.get("estimated_inference_time")
        units = collections_count(prof)
        report["models"][name] = {
            "inference_ms": round(us / 1000, 2) if us else None,
            "peak_memory_bytes": summary.get("estimated_inference_peak_memory"),
            "layers_by_compute_unit": units,
            "compile_job": cj.job_id, "profile_job": pj.job_id,
        }
        print(f"[{name}] {report['models'][name]}")
        if args.runtime == "onnx":
            out = ROOT / "models" / f"{path.stem}.aihub.onnx"
            target.download(str(out))
            print(f"[{name}] AI Hub-optimised model saved to {out}")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "aihub_profile.json").write_text(json.dumps(report, indent=1))
    print("saved docs/aihub_profile.json")


def collections_count(prof: dict) -> dict:
    counts: dict[str, int] = {}
    for layer in prof.get("execution_detail", []) or []:
        unit = layer.get("compute_unit", "?")
        counts[unit] = counts.get(unit, 0) + 1
    return counts


if __name__ == "__main__":
    main()
