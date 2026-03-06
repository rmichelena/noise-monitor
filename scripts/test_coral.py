#!/usr/bin/env python3
"""
Test del Google Coral USB Accelerator.
Verifica carga del delegate Edge TPU, inferencia y rendimiento.

Usa ai-edge-litert con feranick/libedgetpu (compatible Python 3.12+).
No requiere pycoral.
"""

import time
import sys
import os
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate


MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "ssd_mobilenet_v2_coco_edgetpu.tflite"
)
LABELS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "coco_labels.txt"
)


def load_labels(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


def test_coral(model_path=MODEL_PATH, num_inferences=100):
    print("Cargando Edge TPU delegate...")
    try:
        delegate = load_delegate("libedgetpu.so.1", options={"device": "usb"})
    except ValueError as e:
        print(f"ERROR cargando delegate: {e}")
        print("Verifica: libedgetpu instalado, Coral conectado, usuario en grupo plugdev")
        return False

    interpreter = Interpreter(
        model_path=model_path, experimental_delegates=[delegate]
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]

    print(f"Modelo cargado: input {input_shape}, {len(output_details)} outputs")

    dummy = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    interpreter.set_tensor(input_details[0]["index"], dummy)
    interpreter.invoke()  # warmup

    start = time.time()
    for _ in range(num_inferences):
        interpreter.invoke()
    elapsed = time.time() - start

    ms_per_frame = elapsed / num_inferences * 1000
    fps = num_inferences / elapsed
    print(f"Benchmark ({num_inferences} inferencias): {ms_per_frame:.1f} ms/frame, {fps:.1f} FPS")

    # Decodificar última inferencia
    labels = load_labels(LABELS_PATH)
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]
    count = int(interpreter.get_tensor(output_details[3]["index"])[0])

    detections_above_threshold = sum(1 for s in scores if s >= 0.3)
    print(f"Detecciones (score >= 0.3): {detections_above_threshold}")
    for i in range(min(detections_above_threshold, 5)):
        label = labels.get(int(classes[i]), f"id:{int(classes[i])}")
        print(f"  - {label}: {scores[i]:.2f}")

    if fps > 30:
        print("RESULTADO: OK")
        return True
    else:
        print(f"RESULTADO: ADVERTENCIA (FPS bajo: {fps:.1f})")
        return fps > 10


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    success = test_coral(model_path=model)
    sys.exit(0 if success else 1)
