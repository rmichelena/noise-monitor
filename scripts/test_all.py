#!/usr/bin/env python3
"""
Test integrado de los tres componentes del noise monitor:
  1. Cámara USB (OpenCV)
  2. Micrófono Dayton iMM-6C (sounddevice)
  3. Coral USB Accelerator (ai-edge-litert + Edge TPU delegate)

Captura un frame real de la cámara, lo pasa por el Coral para detección,
y simultáneamente graba 1s de audio del micrófono.
"""

import time
import os
import sys
import numpy as np
import cv2
import sounddevice as sd
from ai_edge_litert.interpreter import Interpreter, load_delegate


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "ssd_mobilenet_v2_coco_edgetpu.tflite")
LABELS_PATH = os.path.join(SCRIPT_DIR, "..", "models", "coco_labels.txt")


def load_labels(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


def find_imm6c():
    for i, d in enumerate(sd.query_devices()):
        if "iMM-6C" in d["name"] or "iMM6C" in d["name"]:
            return i
    return None


def main():
    results = {}

    # --- CÁMARA ---
    print("=" * 50)
    print("1. CÁMARA USB")
    print("=" * 50)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, frame = cap.read()
    if ret:
        print(f"   Captura: {frame.shape[1]}x{frame.shape[0]} OK")
        results["camera"] = True
    else:
        print("   ERROR: no se capturó frame")
        results["camera"] = False

    # --- AUDIO ---
    print()
    print("=" * 50)
    print("2. MICRÓFONO iMM-6C")
    print("=" * 50)
    mic_idx = find_imm6c()
    if mic_idx is not None:
        recording = sd.rec(44100, samplerate=44100, channels=1, device=mic_idx, dtype="float32")
        sd.wait()
        rms = np.sqrt(np.mean(recording**2))
        db = 20 * np.log10(rms / 20e-6) if rms > 0 else -np.inf
        print(f"   Dispositivo: [{mic_idx}]")
        print(f"   Grabación 1s: {len(recording)} muestras, ~{db:.1f} dB (sin calibrar)")
        results["audio"] = True
    else:
        print("   ERROR: iMM-6C no encontrado")
        results["audio"] = False

    # --- CORAL TPU ---
    print()
    print("=" * 50)
    print("3. CORAL USB + DETECCIÓN")
    print("=" * 50)
    try:
        delegate = load_delegate("libedgetpu.so.1", options={"device": "usb"})
        interpreter = Interpreter(model_path=MODEL_PATH, experimental_delegates=[delegate])
        interpreter.allocate_tensors()

        labels = load_labels(LABELS_PATH)
        input_details = interpreter.get_input_details()
        input_size = tuple(input_details[0]["shape"][1:3])

        if ret:
            resized = cv2.resize(frame, (input_size[1], input_size[0]))
            interpreter.set_tensor(input_details[0]["index"], np.expand_dims(resized, 0))
        else:
            dummy = np.random.randint(0, 255, input_details[0]["shape"], dtype=np.uint8)
            interpreter.set_tensor(input_details[0]["index"], dummy)

        start = time.time()
        interpreter.invoke()
        ms = (time.time() - start) * 1000

        output_details = interpreter.get_output_details()
        scores = interpreter.get_tensor(output_details[2]["index"])[0]
        classes = interpreter.get_tensor(output_details[1]["index"])[0]
        n_det = sum(1 for s in scores if s >= 0.3)

        print(f"   Inferencia: {ms:.1f} ms")
        print(f"   Detecciones (>= 0.3): {n_det}")
        for i in range(min(n_det, 5)):
            label = labels.get(int(classes[i]), f"id:{int(classes[i])}")
            print(f"     - {label}: {scores[i]:.2f}")
        results["coral"] = True

    except Exception as e:
        print(f"   ERROR: {e}")
        results["coral"] = False

    cap.release()

    # --- RESUMEN ---
    print()
    print("=" * 50)
    print("RESUMEN")
    print("=" * 50)
    all_ok = all(results.values())
    for component, ok in results.items():
        status = "OK" if ok else "FALLO"
        print(f"   {component}: {status}")

    if all_ok:
        print("\n   PIPELINE COMPLETO: TODOS LOS COMPONENTES OK")
    else:
        print("\n   HAY COMPONENTES CON FALLO")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
