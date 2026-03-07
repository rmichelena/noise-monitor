#!/usr/bin/env python3
"""
Noise Monitor - Orquestador principal.
Arranca AudioMonitor, VideoMonitor y EventManager en threads.
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
    """Configura logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main():
    setup_logging()
    logger = logging.getLogger("main")

    base_dir = Path(__file__).parent.resolve()
    config = load_config(base_dir / "config.yaml")

    audio = AudioMonitor(config=config)
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

    # EventManager en thread (consume cola)
    import threading

    t_event = threading.Thread(target=event_mgr.run, daemon=True)
    t_event.start()

    # Video en thread (captura + inferencia)
    t_video = threading.Thread(target=video.run_loop, daemon=True)
    t_video.start()

    import time

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
