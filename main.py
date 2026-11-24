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
from src.perceptual_trigger import Settings
from src.utils import ConfigError, UnmigratedDatabaseError, init_database, load_config, validate_config
from src.video_monitor import VideoMonitor


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def _load_audio_classifier(config: dict, base_dir: Path):
    """Carga el backend configurado de YAMNet."""
    ac_cfg = config.get("audio_classifier")
    if not ac_cfg:
        return None

    model_path = base_dir / ac_cfg["model_path"]
    labels_path = base_dir / ac_cfg["labels_path"]
    backend = ac_cfg.get("backend", "cpu")
    fallback_to_cpu = ac_cfg.get("fallback_to_cpu", True)
    coral_path = base_dir / ac_cfg["coral_model_path"] if "coral_model_path" in ac_cfg else None

    if backend == "coral" and fallback_to_cpu and not model_path.is_file():
        raise ConfigError("el fallback de YAMNet Coral requiere audio_classifier.model_path (modelo CPU)")
    if backend == "cpu" and not model_path.is_file():
        logging.getLogger("main").warning(
            "YAMNet model no encontrado en %s — clasificación deshabilitada. "
            "Ejecutar: scripts/download_yamnet.sh",
            model_path,
        )
        return None

    from src.audio_classifier import AudioClassifier

    try:
        return AudioClassifier(
            model_path=model_path,
            labels_path=labels_path,
            category_map=ac_cfg.get("category_map", {}),
            min_score=float(ac_cfg.get("min_score", 0.3)),
            backend=backend,
            coral_model_path=coral_path,
            fallback_to_cpu=fallback_to_cpu,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        if backend != "coral":
            raise
        raise ConfigError(f"No se pudo inicializar YAMNet Coral: {exc}") from exc


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
    # Validar aquí y no dentro de los threads: una clave ausente reventaba en
    # AudioMonitor con un KeyError sin nombre y systemd reiniciaba en bucle.
    try:
        config = validate_config(load_config(base_dir / "config.yaml"), base_dir=base_dir)
        settings = Settings.load(config)
        if settings is not None:
            try:
                import phonometry  # noqa: F401
            except ImportError as exc:
                raise ConfigError(
                    "criterio psychoacoustic requiere requirements-trigger.txt "
                    "(phonometry) en el entorno de captura"
                ) from exc
        classifier = _load_audio_classifier(config, base_dir)
    except (FileNotFoundError, ConfigError, ValueError) as exc:
        logger.error("Config inválida: %s", exc)
        # Código distinto de 1: systemd solo evita reiniciar este fallo
        # permanente (`RestartPreventExitStatus=78`), no cualquier crash.
        return 78

    # Antes de abrir audio y video. Si lo hiciera recién EventManager, la
    # excepción saldría con código 1 y systemd reiniciaría en bucle.
    try:
        init_database(config, base_dir)
    except UnmigratedDatabaseError as exc:
        logger.error("Base de datos: %s", exc)
        return 78

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
