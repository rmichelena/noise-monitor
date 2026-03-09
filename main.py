#!/usr/bin/env python3
"""
Noise Monitor - Orquestador principal.
Arranca AudioMonitor (con YAMNet), VideoMonitor (con tracker) y EventManager en threads.
"""

import logging
import signal
import sys
from pathlib import Path

from src.audio_monitor import AudioMonitor
from src.event_manager import EventManager
from src.utils import load_config
from src.video_monitor import VideoMonitor


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def _load_audio_classifier(config: dict, base_dir: Path):
    """Intenta cargar AudioClassifier (YAMNet). Retorna None si falta modelo."""
    ac_cfg = config.get("audio_classifier")
    if not ac_cfg:
        return None

    model_path = base_dir / ac_cfg["model_path"]
    labels_path = base_dir / ac_cfg["labels_path"]

    if not model_path.exists():
        logging.getLogger("main").warning(
            "YAMNet model no encontrado en %s — clasificación deshabilitada. "
            "Ejecutar: scripts/download_yamnet.sh",
            model_path,
        )
        return None

    from src.audio_classifier import AudioClassifier

    return AudioClassifier(
        model_path=str(model_path),
        labels_path=str(labels_path),
        category_map=ac_cfg.get("category_map", {}),
        min_score=float(ac_cfg.get("min_score", 0.3)),
    )


def main():
    setup_logging()
    logger = logging.getLogger("main")

    base_dir = Path(__file__).parent.resolve()
    config = load_config(base_dir / "config.yaml")

    classifier = _load_audio_classifier(config, base_dir)

    audio = AudioMonitor(config=config, base_dir=base_dir, audio_classifier=classifier)
    video = VideoMonitor(config=config, base_dir=base_dir)
    event_mgr = EventManager(audio, video, config=config, base_dir=base_dir)

    def shutdown(signum=None, frame=None):
        logger.info("Apagando...")
        audio.stop()
        video.stop()
        event_mgr.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("Iniciando pipeline...")
    audio.start()
    video.start()

    import threading
    import time

    t_event = threading.Thread(target=event_mgr.run, daemon=True)
    t_event.start()

    t_video = threading.Thread(target=video.run_loop, daemon=True)
    t_video.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
