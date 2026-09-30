"""Generate synthetic demo screens + a demo video.

All data is fictional: the bank is invented, the card number is the standard
Visa test number, the AWS key is AWS's own documentation example, and the
Aadhaar-format number is randomly generated with a valid checksum.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from screenveil.classify import verhoeff_check_digit, verhoeff_valid  # noqa: E402

OUT = Path(__file__).resolve().parent / "assets"
W, H = 1280, 720
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


_FALLBACK = {FONT: ["arial.ttf", "Arial.ttf"], BOLD: ["arialbd.ttf", "Arial Bold.ttf"],
             MONO: ["consola.ttf", "cour.ttf", "Menlo.ttc"]}


def font(size, path=FONT):
    for cand in [path] + _FALLBACK.get(path, []):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:  # fonts differ per OS
            continue
    return ImageFont.load_default()


def fake_aadhaar(rng: random.Random) -> str:
    body = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    d = body + verhoeff_check_digit(body)
    assert verhoeff_valid(d)
    return f"{d[:4]} {d[4:8]} {d[8:]}"


def invalid_12(rng: random.Random) -> str:
    while True:
        d = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(11))
        if not verhoeff_valid(d):
            return f"{d[:4]} {d[4:8]} {d[8:]}"


def bank_page(rng) -> Image.Image:
    """A tall netbanking page (scrolled in the video)."""
    img = Image.new("RGB", (W, 1500), (245, 247, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 64], fill=(18, 52, 110))
    d.text((32, 18), "Aurora Bank  ·  NetBanking (demo)", font=font(24, BOLD), fill="white")
    d.text((W - 260, 22), "Logout   Help", font=font(18), fill="white")
    y = 100
    d.text((48, y), "Welcome back, Priya", font=font(28, BOLD), fill=(20, 20, 30))
    y += 60
    card = [(48, y, 620, y + 260), (660, y, 1232, y + 260)]
    for box in card:
        d.rounded_rectangle(box, 14, fill="white", outline=(220, 224, 230))
    rows = [("Savings A/C No.", "5023 0118 7742 61"), ("IFSC", "AURB0001234"),
            ("Branch", "Koramangala, Bengaluru"), ("Available balance", "Rs 1,24,560.00")]
    yy = y + 24
    for k, v in rows:
        d.text((72, yy), k, font=font(18), fill=(110, 116, 130))
        d.text((300, yy), v, font=font(20, BOLD), fill=(20, 20, 30))
        yy += 54
    rows = [("Aadhaar", fake_aadhaar(rng)), ("PAN", "ABCPK1234Z"),
            ("Mobile", "+91 98765 43210"), ("UPI ID", "priya.s@okaxis")]
    yy = y + 24
    for k, v in rows:
        d.text((684, yy), k, font=font(18), fill=(110, 116, 130))
        d.text((860, yy), v, font=font(20, BOLD), fill=(20, 20, 30))
        yy += 54
    y += 300
    d.text((48, y), "Recent transactions", font=font(24, BOLD), fill=(20, 20, 30))
    y += 48
    txns = [("29 Sep", "Swiggy order  ·  Order ID " + invalid_12(rng), "- Rs 412.00"),
            ("28 Sep", "Salary credit  ·  Ref 20260928", "+ Rs 85,000.00"),
            ("27 Sep", "Card payment  ·  4111 1111 1111 1111", "- Rs 2,349.00"),
            ("26 Sep", "Electricity bill  ·  Consumer no. 7702", "- Rs 1,180.00"),
            ("25 Sep", "UPI to rahul.k@ybl", "- Rs 600.00"),
            ("24 Sep", "Interest credit  ·  Q2 2026", "+ Rs 312.40")]
    for dt, desc, amt in txns:
        d.rounded_rectangle([48, y, 1232, y + 58], 10, fill="white", outline=(228, 231, 236))
        d.text((72, y + 17), dt, font=font(18), fill=(110, 116, 130))
        d.text((190, y + 16), desc, font=font(19), fill=(20, 20, 30))
        d.text((1040, y + 16), amt, font=font(19, BOLD), fill=(20, 20, 30))
        y += 72
    y += 20
    d.text((48, y), "Registered email: priya.sharma@example.com", font=font(19), fill=(60, 64, 80))
    y += 50
    d.text((48, y), "Need help? Call 1800 000 000 (toll free, demo)", font=font(18), fill=(110, 116, 130))
    return img


def code_screen(rng) -> Image.Image:
    img = Image.new("RGB", (W, H), (30, 32, 38))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 36], fill=(44, 47, 56))
    d.text((16, 8), "settings.py  —  payments-service  (demo)", font=font(16), fill=(200, 204, 214))
    token = "ghp_" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                             for _ in range(36))
    lines = [
        ("# deployment settings", (120, 128, 140)),
        ("import os", (198, 120, 221)),
        ("", None),
        ("REGION = \"ap-south-1\"", (220, 220, 220)),
        ("TIMEOUT_SECONDS = 30", (220, 220, 220)),
        ("AWS_ACCESS_KEY_ID = \"AKIAIOSFODNN7EXAMPLE\"", (220, 220, 220)),
        (f"GITHUB_TOKEN = \"{token}\"", (220, 220, 220)),
        ("DATABASE_URL = \"postgres://admin:S3cretPass@db.internal:5432/prod\"", (220, 220, 220)),
        ("MAX_RETRIES = 5", (220, 220, 220)),
        ("", None),
        ("def handler(event):", (97, 175, 239)),
        ("    return {\"status\": 200}", (220, 220, 220)),
    ]
    y = 60
    for i, (t, c) in enumerate(lines, 1):
        d.text((16, y), f"{i:>2}", font=font(18, MONO), fill=(90, 96, 110))
        if t:
            d.text((64, y), t, font=font(18, MONO), fill=c)
        y += 34
    return img


def chat_screen(with_toast: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), (236, 229, 221))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 360, H], fill="white")
    d.text((24, 24), "Chats (demo)", font=font(24, BOLD), fill=(20, 20, 30))
    for i, (n, m) in enumerate([("Team Alpha", "standup at 10"), ("Mom", "call me"), ("Ankit", "sent a photo")]):
        d.text((24, 90 + i * 72), n, font=font(20, BOLD), fill=(20, 20, 30))
        d.text((24, 118 + i * 72), m, font=font(17), fill=(110, 116, 130))
    msgs = [("Hi team, quick update below", False), ("Project Falcon launch moved to Friday", False),
            ("Noted, will update the deck", True), ("Can you share the numbers in the call?", False)]
    y = 90
    for t, me in msgs:
        tw = int(d.textlength(t, font=font(20))) + 36
        x = W - tw - 40 if me else 400
        d.rounded_rectangle([x, y, x + tw, y + 48], 12, fill=(217, 253, 211) if me else "white")
        d.text((x + 18, y + 12), t, font=font(20), fill=(20, 20, 30))
        y += 70
    if with_toast:
        x0, y0 = W - 470, H - 170
        d.rounded_rectangle([x0, y0, W - 20, H - 24], 14, fill=(38, 40, 48))
        d.text((x0 + 20, y0 + 16), "Messages  ·  now", font=font(16), fill=(170, 176, 190))
        d.text((x0 + 20, y0 + 50), "Your OTP for login is 482913.", font=font(20, BOLD), fill="white")
        d.text((x0 + 20, y0 + 86), "Do not share it with anyone.", font=font(18), fill=(210, 214, 224))
    return img


def to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


SRC = OUT / "src"


def source_screens() -> dict[str, np.ndarray]:
    """The four source screens as BGR images.

    Loaded from demo/assets/src/*.png when present, so every machine gets
    pixel-identical input (fonts differ between Windows, macOS and Linux,
    which would change OCR results). Generated and saved otherwise.
    """
    names = ["bank_full", "code", "chat_plain", "chat_toast"]
    if all((SRC / f"{n}.png").exists() for n in names):
        return {n: cv2.imread(str(SRC / f"{n}.png")) for n in names}
    rng = random.Random(7)
    imgs = {"bank_full": to_bgr(bank_page(rng)), "code": to_bgr(code_screen(rng)),
            "chat_plain": to_bgr(chat_screen(False)), "chat_toast": to_bgr(chat_screen(True))}
    SRC.mkdir(parents=True, exist_ok=True)
    for n, im in imgs.items():
        cv2.imwrite(str(SRC / f"{n}.png"), im)
    return imgs


def demo_frames():
    """Frames of the demo sequence (lossless, like a real screen capture)."""
    s = source_screens()
    bank_bgr = s["bank_full"]
    frames = [bank_bgr[:H]] * 20                              # read the page
    for i in range(1, 31):                                     # smooth scroll down 600 px
        off = int(600 * (1 - np.cos(np.pi * i / 30)) / 2)
        frames.append(bank_bgr[off:off + H])
    frames += [bank_bgr[600:600 + H]] * 15
    frames += [s["chat_plain"]] * 12 + [s["chat_toast"]] * 25  # OTP notification pops up
    frames += [s["code"]] * 25                                 # switch to the code editor
    return [np.ascontiguousarray(f) for f in frames]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = source_screens()
    cv2.imwrite(str(OUT / "bank.png"), s["bank_full"][:H])
    cv2.imwrite(str(OUT / "code.png"), s["code"])
    cv2.imwrite(str(OUT / "chat.png"), s["chat_toast"])
    frames = demo_frames()
    vw = cv2.VideoWriter(str(OUT / "demo_input.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"wrote {len(frames)} frames to {OUT/'demo_input.mp4'}")


if __name__ == "__main__":
    main()
