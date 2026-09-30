"""Detection check on all demo content: 14 planted sensitive items, plus
look-alikes that must stay visible (order ID failing the Aadhaar checksum,
balances, dates, reference numbers, a toll-free number)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))
import make_demo as M  # noqa: E402
from screenveil.backends.tesseract_engine import TesseractTextEngine  # noqa: E402
from screenveil.classify import ClassifierConfig, SensitiveTextClassifier  # noqa: E402

EXPECTED = {"bank": {"EMAIL", "CARD", "UPI_ID", "IFSC", "BANK_ACCOUNT", "PAN", "AADHAAR", "PHONE"},
            "code": {"AWS_KEY", "GITHUB_TOKEN", "CONN_STRING"},
            "chat": {"CUSTOM_TERM", "OTP"}}
EXPECTED_COUNT = 14   # bank has two UPI IDs

_s = M.source_screens()
screens = {"bank": _s["bank_full"], "code": _s["code"], "chat": _s["chat_toast"]}
ocr = TesseractTextEngine()
clf = SensitiveTextClassifier(ClassifierConfig(custom_terms=["Project Falcon"]))
total, ok = 0, True
for name, img in screens.items():
    found = clf.classify_lines(ocr.read_lines(img))
    kinds = {f.kind for f in found}
    total += len(found)
    missing, extra = EXPECTED[name] - kinds, kinds - EXPECTED[name]
    ok &= not missing and not extra
    print(f"{name:5s} found {len(found):2d}  missing={sorted(missing)}  unexpected={sorted(extra)}")
print(f"total {total}/{EXPECTED_COUNT}", "PASS" if ok and total == EXPECTED_COUNT else "FAIL")
