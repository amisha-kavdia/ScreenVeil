"""Export the OCR models ScreenVeil runs on the Snapdragon NPU.

Produces, with *static* shapes (the Hexagon NPU needs them):
  models/easyocr_detector.onnx    CRAFT text detector, input 1x3xHxW (default 256x1920)
  models/easyocr_recognizer.onnx  CRNN/CTC recogniser, input 1x1x64xW (default 64x800)

Two sources are supported:

  --source aihub   Qualcomm AI Hub Models (`pip install "qai-hub-models[easyocr]"`).
                   Uses the AI Hub build of EasyOCR, then re-exports each
                   component at ScreenVeil's band shape.
  --source easyocr The open-source `easyocr` package (`pip install easyocr`).

Run this on any PC with PyTorch (x64 is fine); copy the two .onnx files into
`models/` on the Snapdragon laptop. Then optionally:
  python tools/aihub_profile.py     # compile + profile on a real Snapdragon X Elite in the cloud
  python tools/quantize.py          # INT8/INT16 QDQ model for the NPU's integer engine
"""
from __future__ import annotations

import argparse
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "models"


def load_components(source: str):
    """Return (detector nn.Module, recognizer nn.Module) in eval mode."""
    if source == "aihub":
        # AI Hub wraps EasyOCR; its collection model exposes the two stages.
        from qai_hub_models.models.easyocr import Model  # type: ignore
        m = Model.from_pretrained()
        det = getattr(m, "detector", None) or m.components["detector"]
        rec = getattr(m, "recognizer", None) or m.components["recognizer"]
        return det.eval(), rec.eval()
    import easyocr  # type: ignore
    reader = easyocr.Reader(["en"], gpu=False)
    det = reader.detector.module if hasattr(reader.detector, "module") else reader.detector
    rec = reader.recognizer.module if hasattr(reader.recognizer, "module") else reader.recognizer
    return det.eval(), rec.eval()


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["aihub", "easyocr"], default="aihub")
    ap.add_argument("--det-shape", default="256x1920", help="HxW of one screen band")
    ap.add_argument("--rec-width", type=int, default=800)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    det, rec = load_components(args.source)
    h, w = (int(v) for v in args.det_shape.split("x"))

    class Det(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            y = self.m(x)
            return y[0] if isinstance(y, (tuple, list)) else y   # (1, H/2, W/2, 2) score maps

    class Rec(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            try:
                return self.m(x)
            except TypeError:                     # EasyOCR's CTC model takes an unused `text` arg
                return self.m(x, None)

    with torch.no_grad():
        torch.onnx.export(Det(det), torch.randn(1, 3, h, w), OUT / "easyocr_detector.onnx",
                          input_names=["image"], output_names=["scores"], opset_version=17)
        torch.onnx.export(Rec(rec), torch.randn(1, 1, 64, args.rec_width),
                          OUT / "easyocr_recognizer.onnx", input_names=["crop"],
                          output_names=["logits"], opset_version=17)
    print("wrote", OUT / "easyocr_detector.onnx", "and", OUT / "easyocr_recognizer.onnx")
    print("next: python tools/aihub_profile.py  (real Snapdragon X Elite latency)")


if __name__ == "__main__":
    main()
