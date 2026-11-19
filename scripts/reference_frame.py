#!/usr/bin/env python3
"""
Captura un frame con las líneas de referencia de la escala de pista y lo
publica en Buzz para que el dueño confirme las coordenadas.

Pausa el pipeline mientras usa la cámara (V4L2 no admite dos lectores) y lo
vuelve a arrancar siempre, incluso si la captura falla.

Uso: python scripts/reference_frame.py --post
"""

import argparse
import os
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.utils import load_config  # noqa: E402

CHANNEL = "d113eafd-6fc3-4477-80ed-f08b3fc77770"
THREAD = "1a4293bc8da57fc715c45af48d4cf0aec189fff8b9afbe52690f2181d6efab1c"
FRAME_PATH = Path("/tmp/reference_frame.png")


def _strip_png(src: Path, dst: Path) -> None:
    """Deja solo IHDR/IDAT/IEND: el relay rechaza metadatos no canónicos."""
    data = src.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        dst.write_bytes(data)
        return
    out = bytearray(data[:8])
    pos = 8
    keep = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS"}
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        if chunk_type in keep:
            out += data[pos:pos + 12 + length]
        pos += 12 + length
    dst.write_bytes(out)


def capture() -> tuple[bool, str]:
    config = load_config(PROJECT_DIR / "config.yaml")
    left = int(config["speed"]["reference_left_x"])
    right = int(config["speed"]["reference_right_x"])
    meters = float(config["speed"]["reference_distance_m"])

    subprocess.run(["sudo", "systemctl", "stop", "noise-monitor"], check=False)
    time.sleep(2)
    ok = False
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        frame = None
        for _ in range(30):
            ret, f = cap.read()
            if ret:
                frame = f
            time.sleep(0.05)
        cap.release()
        if frame is None:
            return False, "no se pudo leer ningún frame"
        h = frame.shape[0]
        for x in (left, right):
            cv2.line(frame, (x, 0), (x, h), (0, 0, 255), 6)
            cv2.putText(frame, f"x={x}", (x + 10 if x < 1800 else x - 130, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        raw = Path("/tmp/reference_frame_raw.png")
        cv2.imwrite(str(raw), frame)
        _strip_png(raw, FRAME_PATH)
        raw.unlink(missing_ok=True)
        ok = True
        return True, f"{meters:.0f} m entre x={left} y x={right}"
    finally:
        subprocess.run(["sudo", "systemctl", "start", "noise-monitor"], check=False)
        if not ok:
            FRAME_PATH.unlink(missing_ok=True)


def post(text: str) -> int:
    cmd = [
        "/home/roberto/buzz/bin/buzz", "messages", "send",
        "--channel", CHANNEL, "--reply-to", THREAD,
        "--file", str(FRAME_PATH), "--content", text,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if result.returncode != 0:
        print(f"ERROR publicando: {result.stderr.strip()}", file=sys.stderr)
    else:
        print(result.stdout.strip())
    return result.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true")
    args = ap.parse_args()

    if not FRAME_PATH.exists():
        ok, detail = capture()
        if not ok:
            print(f"captura fallida: {detail}")
            return 1
    else:
        detail = "frame existente"

    if args.post:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        text = (
            f"Frame de referencia de las {stamp} ({detail}).\n\n"
            "Las dos líneas rojas marcan `reference_left_x` y `reference_right_x`, "
            "que uso como los 30 m de pista. Si no caen justo sobre los árboles, "
            "dime las X correctas (o mándame un frame) y ajusto `config.yaml`."
        )
        return post(text)
    print(f"frame listo en {FRAME_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
