#!/bin/bash
# Descarga videos de tráfico con audio para desarrollo sin balcón (Fase 0.5)
# Requiere: yt-dlp, ffmpeg
#
# Uso:
#   ./scripts/download_test_videos.sh [URL1] [URL2] ...
#   O sin args: descarga URLs por defecto (editar el array VIDEO_URLS abajo)
#
# Criterios de selección:
#   - Ángulo similar al balcón (vista cenital o semi-cenital de calle)
#   - Audio de tráfico real (motores, bocinas)
#   - Variedad: día, noche, lluvia si hay

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEOS_DIR="$PROJECT_DIR/data/test_videos"

mkdir -p "$VIDEOS_DIR"
cd "$VIDEOS_DIR"

# URLs - pasar como argumentos o editar aquí
# Buscar: "traffic camera live", "urban traffic noise", "street traffic 4k"
VIDEO_URLS=()

if [ $# -gt 0 ]; then
    VIDEO_URLS=("$@")
fi

if [ ${#VIDEO_URLS[@]} -eq 0 ]; then
    echo "Edita este script y agrega URLs de videos de tráfico con audio."
    echo "Busca en YouTube: 'traffic camera live', 'urban traffic noise', 'street traffic 4k'"
    echo ""
    echo "Ejemplo de uso con URLs:"
    echo "  ./scripts/download_test_videos.sh 'https://youtube.com/watch?v=xxx' 'https://youtube.com/watch?v=yyy'"
    exit 1
fi

echo "Descargando a $VIDEOS_DIR ..."

for url in "${VIDEO_URLS[@]}"; do
    echo "--- Descargando: $url"
    # H.264 (avc1) obligatorio: OpenCV no decodifica AV1 en muchos sistemas
    # 136+140 = 720p H.264, fallback 18 = 360p merged
    yt-dlp -f "136+140/18" \
        --merge-output-format mp4 \
        -o "traffic_%(id)s.%(ext)s" \
        "$url"
done

echo ""
echo "Verificando audio en los videos:"
for f in *.mp4; do
    if [ -f "$f" ]; then
        echo "  $f:"
        ffprobe -i "$f" 2>&1 | grep -E "Stream|Audio" || true
    fi
done

echo ""
echo "Listo. Para extraer audio de un video:"
echo "  ffmpeg -i traffic_xxx.mp4 -vn -ar 44100 -ac 1 -f wav traffic_xxx_audio.wav"
