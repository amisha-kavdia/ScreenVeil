"""Measure real power draw per compute unit on *your* laptop (Windows, on battery).

The NPU Conductor ranks devices by latency x watts. Out of the box it uses
modelled watts; this script replaces them with measurements:

  1. Unplug the charger, close other apps, set Windows to "Best performance".
  2. python tools/calibrate_power.py --seconds 60
  3. python -m screenveil run --power-profile docs/power_profile.json

For each device it records the battery discharge rate (Windows WMI
BatteryStatus.DischargeRate) while idle, then while running the text model
continuously; active watts = loaded - idle.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def discharge_w() -> float | None:
    import subprocess
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus | "
                          "Measure-Object -Property DischargeRate -Sum).Sum"],
                         capture_output=True, text=True).stdout.strip()
    return float(out) / 1000 if out else None


def average_w(seconds: float, work=None) -> float:
    samples, end = [], time.time() + seconds
    nxt = time.time()
    while time.time() < end:
        if work:
            work()
        if time.time() >= nxt:
            w = discharge_w()
            if w:
                samples.append(w)
            nxt = time.time() + 2
    return sum(samples) / max(1, len(samples))


def main():
    from screenveil.backends.onnx_ocr import OnnxOcrEngine
    from screenveil.backends.ort_session import available_devices
    from screenveil.cli import DET, REC

    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    args = ap.parse_args()
    if sys.platform != "win32":
        raise SystemExit("Windows only (reads battery discharge via WMI)")
    band = cv2.imread(str(ROOT / "demo" / "assets" / "bank.png"))[:256]
    idle = average_w(args.seconds)
    print(f"idle: {idle:.2f} W")
    profile = {}
    for dev in available_devices():
        eng = OnnxOcrEngine(str(DET), str(REC), device=dev)
        loaded = average_w(args.seconds, lambda: eng.read_lines(band))
        profile[dev] = round(max(0.1, loaded - idle), 2)
        print(f"{dev}: {loaded:.2f} W loaded -> {profile[dev]} W active")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "power_profile.json").write_text(json.dumps(profile, indent=1))
    print("saved docs/power_profile.json")


if __name__ == "__main__":
    main()
