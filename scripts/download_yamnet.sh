#!/usr/bin/env bash
# Download YAMNet TFLite model + labels to models/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${1:-$PROJECT_DIR/models}"

mkdir -p "$MODELS_DIR"

YAMNET_URL="https://storage.googleapis.com/audioset/yamnet.tflite"
LABELS_URL="https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

download() {
    local url="$1" path="$2"
    if [ ! -s "$path" ]; then
        curl -fL -o "$path.part" "$url"
        mv "$path.part" "$path"
    fi
}

download "$YAMNET_URL" "$MODELS_DIR/yamnet.tflite"
download "$LABELS_URL" "$MODELS_DIR/yamnet_class_map.csv"

# Las comas pueden ser parte de display_name.
python3 - "$MODELS_DIR" <<'PY'
import csv
import sys
from pathlib import Path

directory = Path(sys.argv[1])
with (directory / "yamnet_class_map.csv").open(encoding="utf-8", newline="") as file:
    classes = list(csv.DictReader(file))
if [int(row["index"]) for row in classes] != list(range(521)):
    raise SystemExit("El mapa de YAMNet debe contener las 521 etiquetas en orden")
(directory / "yamnet_labels.txt").write_text(
    "".join(row["display_name"] + "\n" for row in classes), encoding="utf-8"
)
PY
echo "Listo. Modelos en: $MODELS_DIR"
