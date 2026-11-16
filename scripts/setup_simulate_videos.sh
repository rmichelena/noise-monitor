#!/bin/bash
# Extrae audio de los videos en data/test_videos/ y muestra la config para modo simulado.
# Ejecutar en el server después de que yt-dlp termine las descargas.
#
# Uso: ./scripts/setup_simulate_videos.sh [video_principal]
#   video_principal: nombre del .mp4 a usar (ej: traffic_9wbA9FVtWF4.mp4)
#   Si no se especifica, usa el primero que encuentre.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEOS_DIR="$PROJECT_DIR/data/test_videos"

cd "$VIDEOS_DIR"

# Esperar a que no haya .part (descarga en curso)
while ls *.part 2>/dev/null; do
    echo "Esperando que termine la descarga..."
    sleep 10
done

# Buscar videos completos
VIDEOS=()
for f in traffic_*.mp4; do
    [ -f "$f" ] && VIDEOS+=("$f")
done

if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "No hay videos .mp4 en $VIDEOS_DIR"
    exit 1
fi

# Video a usar
if [ -n "$1" ]; then
    MAIN_VIDEO="$1"
else
    MAIN_VIDEO="${VIDEOS[0]}"
fi

if [ ! -f "$MAIN_VIDEO" ]; then
    echo "Video no encontrado: $MAIN_VIDEO"
    echo "Disponibles: ${VIDEOS[*]}"
    exit 1
fi

# Extraer audio de todos los que no lo tengan
for v in "${VIDEOS[@]}"; do
    base="${v%.mp4}"
    wav="${base}_audio.wav"
    if [ ! -f "$wav" ]; then
        echo "Extrayendo audio: $v -> $wav"
        ffmpeg -y -i "$v" -vn -ar 44100 -ac 1 -f wav "$wav" 2>/dev/null
    else
        echo "Audio ya existe: $wav"
    fi
done

echo ""
echo "=== Configuración para config.yaml (modo simulado) ==="
echo ""
echo "video:"
echo "  source: \"data/test_videos/$MAIN_VIDEO\""
echo ""
echo "debug:"
echo "  simulate: true"
echo "  loop_video: true"
echo "  audio_from_video: true"
echo ""
echo "Ejecutar: python main.py"
