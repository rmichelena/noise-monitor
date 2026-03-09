#!/bin/bash
# Extrae audio de un video MP4 a WAV 44100 Hz mono (formato del pipeline)
# Uso: ./scripts/extract_audio_from_video.sh data/test_videos/traffic_xxx.mp4

set -e

if [ $# -lt 1 ]; then
    echo "Uso: $0 <video.mp4> [output.wav]"
    echo "  Si no se especifica output, usa el mismo nombre con _audio.wav"
    exit 1
fi

INPUT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$INPUT" ]; then
    # Intentar como ruta relativa al proyecto
    INPUT="$PROJECT_DIR/$1"
fi

if [ ! -f "$INPUT" ]; then
    echo "No se encontró: $1"
    exit 1
fi

BASENAME="${INPUT%.*}"
OUTPUT="${2:-${BASENAME}_audio.wav}"

echo "Extrayendo audio: $INPUT -> $OUTPUT"
ffmpeg -y -i "$INPUT" -vn -ar 44100 -ac 1 -f wav "$OUTPUT"

echo "Listo. Verificar en config.yaml:"
echo "  audio:"
echo "    source: file"
echo "    file_path: \"$OUTPUT\""
