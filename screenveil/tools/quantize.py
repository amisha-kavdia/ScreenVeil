"""Quantise the OCR models for the Hexagon NPU's integer engine (QDQ format).

Calibrates on real screen content — the demo screens plus any screenshots you
drop into `calib/` — cut into the same bands ScreenVeil analyses at run time.
Uses ONNX Runtime's QNN-specific quantisation config (uint16 activations,
uint8 weights), the combination recommended for accuracy on the HTP.

  pip install onnx onnxruntime
  python tools/quantize.py
  -> models/easyocr_detector.qdq.onnx, models/easyocr_recognizer.qdq.onnx
  then: set SCREENVEIL_MODELS or rename to use them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from screenveil.backends.onnx_ocr import letterbox, prepare_rec_crop  # noqa: E402


def calib_images() -> list[np.ndarray]:
    paths = sorted((ROOT / "demo" / "assets").glob("before_*.png")) + \
        sorted((ROOT / "demo" / "assets").glob("*.png"))[:6] + sorted((ROOT / "calib").glob("*.png"))
    return [cv2.imread(str(p)) for p in dict.fromkeys(paths)]


def main():
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

    class Reader(CalibrationDataReader):
        def __init__(self, items):
            self.it = iter(items)

        def get_next(self):
            return next(self.it, None)

    imgs = calib_images()
    for kind in ("detector", "recognizer"):
        src = ROOT / "models" / f"easyocr_{kind}.onnx"
        pre = ROOT / "models" / f"easyocr_{kind}.pre.onnx"
        dst = ROOT / "models" / f"easyocr_{kind}.qdq.onnx"
        s = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        inp = s.get_inputs()[0]
        items = []
        if kind == "detector":
            H, W = inp.shape[2], inp.shape[3]
            mean = np.array([0.485, 0.456, 0.406], np.float32) * 255
            std = np.array([0.229, 0.224, 0.225], np.float32) * 255
            for im in imgs:
                for y0 in range(0, max(1, im.shape[0] - H + 1), H // 2):
                    b, _ = letterbox(cv2.cvtColor(im[y0:y0 + H], cv2.COLOR_BGR2RGB), H, W)
                    items.append({inp.name: ((b.astype(np.float32) - mean) / std).transpose(2, 0, 1)[None]})
        else:
            H, W = inp.shape[2], inp.shape[3]
            for im in imgs:
                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                for y0 in range(0, g.shape[0] - 24, 37):          # strips roughly one text line tall
                    items.append({inp.name: prepare_rec_crop(g[y0:y0 + 24, 0:min(g.shape[1], 600)], H, W)})
        print(f"{kind}: {len(items)} calibration samples")
        changed = qnn_preprocess_model(str(src), str(pre))
        cfg = get_qnn_qdq_config(str(pre if changed else src), Reader(items),
                                 activation_type=QuantType.QUInt16, weight_type=QuantType.QUInt8)
        quantize(str(pre if changed else src), str(dst), cfg)
        print("wrote", dst)


if __name__ == "__main__":
    main()
