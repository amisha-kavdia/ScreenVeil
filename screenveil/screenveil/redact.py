"""Compositor: turns the live frame + screen state into the shareable frame.

Text is *painted over*, never blurred: Gaussian blur and light pixelation of
text can be partially reversed, a solid fill cannot. Faces get a heavy
mosaic. Unverified regions get a coarse mosaic so the audience still sees
layout and motion while analysis catches up.
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import Finding

LABELS = {
    "AADHAAR": "Aadhaar", "PAN": "PAN", "CARD": "Card no.", "IFSC": "IFSC",
    "BANK_ACCOUNT": "Account no.", "UPI_ID": "UPI ID", "PHONE": "Phone", "OTP": "OTP",
    "CVV": "CVV", "EMAIL": "Email", "AWS_KEY": "AWS key", "GITHUB_TOKEN": "GitHub token",
    "GOOGLE_KEY": "API key", "SLACK_TOKEN": "Slack token", "STRIPE_KEY": "Stripe key",
    "API_KEY": "API key", "JWT": "Token", "PRIVATE_KEY": "Private key",
    "CONN_STRING": "Password", "CREDENTIAL": "Secret", "CUSTOM_TERM": "Confidential",
    "HIGH_ENTROPY_SECRET": "Secret", "FACE": "",
}


def mosaic(img: np.ndarray, block: int) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def compose(frame: np.ndarray, unverified: np.ndarray, cell: int, findings: list[Finding],
            show_labels: bool = True, pending_block: int = 14,
            fill=(52, 42, 38), text_color=(235, 235, 235)) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()

    if unverified.any():
        mask = cv2.resize(unverified.astype(np.uint8), (unverified.shape[1] * cell,
                          unverified.shape[0] * cell), interpolation=cv2.INTER_NEAREST)[:h, :w]
        m = mask.astype(bool)
        out[m] = mosaic(frame, pending_block)[m]

    for f in findings:
        b = f.box.clipped(w, h)
        if b is None:
            continue
        roi = out[b.y:b.y2, b.x:b.x2]
        if f.kind == "FACE":
            out[b.y:b.y2, b.x:b.x2] = mosaic(frame[b.y:b.y2, b.x:b.x2], max(6, b.w // 6))
            continue
        # dark themes: use a lighter fill so the redaction is visibly deliberate
        roi[:] = fill if roi.mean() > 90 else (92, 86, 82)
        label = LABELS.get(f.kind, "Hidden")
        if show_labels and label and b.h >= 12:
            scale = min(0.5, b.h / 36)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            if tw + 8 <= b.w:
                cv2.putText(out, label, (b.x + 4, b.y + (b.h + th) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)
    return out


def full_shield(frame: np.ndarray, message: str = "ScreenVeil: protection paused") -> np.ndarray:
    """Fail-closed output used when the pipeline itself is unhealthy."""
    out = mosaic(frame, 40)
    out = (out * 0.35).astype(np.uint8)
    h, w = out.shape[:2]
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(out, message, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (240, 240, 240), 2, cv2.LINE_AA)
    return out
