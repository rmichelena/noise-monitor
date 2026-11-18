#!/usr/bin/env python3
"""
Noise Monitor - Orquestador principal.
Arranca AudioMonitor (con YAMNet), VideoMonitor (con tracker) y EventManager en threads.
"""

import logging
import os
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


def _start_web(config: dict, base_dir: Path, audio, video, logger):
    """Arranca el dashboard en un thread. Sin dependencias, sigue sin web."""
    web_cfg = config.get("web") or {}
    if not web_cfg.get("enabled", True):
        logger.info("Dashboard deshabilitado en config")
        return None

    try:
        import threading

        import uvicorn

        from web.app import create_app
    except ImportError as exc:
        logger.error("Dashboard no disponible (falta dependencia): %s", exc)
        return None

    host = web_cfg.get("host", "0.0.0.0")
    port = int(web_cfg.get("port", 8000))
    app = create_app(
        config=config, base_dir=base_dir, audio_monitor=audio, video_monitor=video
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True, name="web").start()
    logger.info("Dashboard disponible en http://%s:%d", host, port)
    return server


def main():
    setup_logging()
    logger = logging.getLogger("main")

    base_dir = Path(__file__).parent.resolve()
    config = load_config(base_dir / "config.yaml")

    classifier = _load_audio_classifier(config, base_dir)

    audio = AudioMonitor(config=config, base_dir=base_dir, audio_classifier=classifier)
    video = VideoMonitor(config=config, base_dir=base_dir)
    event_mgr = EventManager(audio, video, config=config, base_dir=base_dir)

    stopped = False
    web_server = None

    def shutdown(signum=None, frame=None):
        # Handler de señal y bloque finally convergen aquí; sin el guard el
        # segundo stop libera cv2/PortAudio dos veces y el proceso muere por
        # SIGSEGV (visible como core-dump en systemd). os._exit evita además
        # el desmontaje de ~2 GB de frames al final.
        nonlocal stopped
        if stopped:
            return
        stopped = True
        logger.info("Apagando...")
        if web_server:
            web_server.should_exit = True
        audio.stop()
        video.stop()
        event_mgr.stop()
        logging.shutdown()
        os._exit(0)

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

    web_server = _start_web(config, base_dir, audio, video, logger)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
