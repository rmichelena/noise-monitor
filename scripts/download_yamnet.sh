#!/usr/bin/env bash
# Download YAMNet TFLite model + labels to models/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WITH_CORAL=false
if [[ "${1:-}" == "--coral" ]]; then
    WITH_CORAL=true
    shift
fi
MODELS_DIR="${1:-$PROJECT_DIR/models}"

mkdir -p "$MODELS_DIR"

YAMNET_URL="https://storage.googleapis.com/audioset/yamnet.tflite"
CORAL_URL="https://tfhub.dev/google/coral-model/yamnet/classification/coral/1?coral-format=tflite"
CORAL_SHA256="92fa7ad246d585aaef2b4f430524a3e3fb5888ce94f68ebe3d5d2ebad23b9220"
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
if $WITH_CORAL; then
    coral_model="$MODELS_DIR/yamnet_edgetpu.tflite"
    if [[ ! -s "$coral_model" ]]; then
        curl -fL -o "$coral_model.part" "$CORAL_URL"
        if ! printf '%s  %s\n' "$CORAL_SHA256" "$coral_model.part" | sha256sum --check --status; then
            echo "SHA-256 inesperado en la descarga de YAMNet Coral" >&2
            exit 1
        fi
        mv "$coral_model.part" "$coral_model"
    fi
    if ! printf '%s  %s\n' "$CORAL_SHA256" "$coral_model" | sha256sum --check --status; then
        echo "SHA-256 inesperado para $coral_model" >&2
        exit 1
    fi
fi

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
