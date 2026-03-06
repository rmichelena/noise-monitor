#!/usr/bin/env python3
"""
Test de cámara USB (ELP-USBFHD08S-L29).
Verifica captura, resolución, FPS real y estabilidad.
"""

import time
import sys
import cv2
import numpy as np


def test_camera(device_index=0, width=1920, height=1080, fps=30, num_frames=90):
    cap = cv2.VideoCapture(device_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir cámara en índice {device_index}")
        return False

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Cámara abierta: {actual_w}x{actual_h} @ {actual_fps} FPS (reportado)")

    ok_count = 0
    fail_count = 0
    start = time.time()

    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            ok_count += 1
        else:
            fail_count += 1

    elapsed = time.time() - start
    real_fps = ok_count / elapsed if elapsed > 0 else 0

    cap.release()

    print(f"Frames capturados: {ok_count}/{num_frames} ({fail_count} fallidos)")
    print(f"FPS real: {real_fps:.1f}")
    print(f"Tiempo total: {elapsed:.2f}s")

    if fail_count == 0 and real_fps > 20:
        print("RESULTADO: OK")
        return True
    else:
        print("RESULTADO: FALLO")
        return False


if __name__ == "__main__":
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    success = test_camera(device_index=device)
    sys.exit(0 if success else 1)
