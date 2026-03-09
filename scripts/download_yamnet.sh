#!/usr/bin/env bash
# Download YAMNet TFLite model + labels to models/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

mkdir -p "$MODELS_DIR"

YAMNET_URL="https://storage.googleapis.com/audioset/yamnet.tflite"
LABELS_URL="https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

echo "Descargando yamnet.tflite..."
if [ ! -f "$MODELS_DIR/yamnet.tflite" ]; then
    curl -L -o "$MODELS_DIR/yamnet.tflite" "$YAMNET_URL"
    echo "  → yamnet.tflite OK ($(du -h "$MODELS_DIR/yamnet.tflite" | cut -f1))"
else
    echo "  → yamnet.tflite ya existe, omitiendo"
fi

echo "Descargando yamnet_class_map.csv y generando yamnet_labels.txt..."
if [ ! -f "$MODELS_DIR/yamnet_labels.txt" ]; then
    curl -L -o "$MODELS_DIR/yamnet_class_map.csv" "$LABELS_URL"
    # CSV format: index,mid,display_name — extract display_name (col 3)
    tail -n +2 "$MODELS_DIR/yamnet_class_map.csv" | cut -d',' -f3 | sed 's/^"//;s/"$//' > "$MODELS_DIR/yamnet_labels.txt"
    echo "  → yamnet_labels.txt OK ($(wc -l < "$MODELS_DIR/yamnet_labels.txt") clases)"
else
    echo "  → yamnet_labels.txt ya existe, omitiendo"
fi

echo "Listo. Modelos en: $MODELS_DIR"
