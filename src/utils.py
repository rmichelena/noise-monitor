"""
Utilidades base para el sistema de monitoreo de ruido.
- Filtro A-weighting IEC 61672-1
- Conversión RMS → dB
- Carga de configuración
- Inicialización de SQLite
"""

import logging
import math
import os
import sqlite3
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import bilinear, lfilter

logger_module = logging.getLogger(__name__)

# Presión de referencia estándar (20 µPa)
REFERENCE_PRESSURE = 20e-6
SPEECH_CLASSES = {
    "Speech", "Child speech", "Child speech, kid speaking", "Conversation",
    "Narration", "Narration, monologue",
    "Babbling", "Speech synthesizer", "Whispering",
}

# Nivel mínimo representable. Un bloque de silencio digital (RMS exactamente
# 0) no tiene nivel: devolver -inf contamina SQLite, el JSON del dashboard
# (`-Infinity` no es JSON válido) y la barra quemada en los clips. Se acota a
# este piso, que queda por debajo del ruido de fondo de cualquier ADC real.
MIN_DB = -120.0

# Coeficientes del filtro A-weighting (precalculados por sample rate)
_a_weight_cache = {}


def design_a_weighting(fs: float):
    """
    Diseña filtro A-weighting según IEC 61672-1.
    Retorna coeficientes (b, a) para lfilter (digital).

    La normalización del estándar busca A(1000 Hz) = 0 dB: el factor es
    10^(+2/20) ≈ 1.2589. Con el signo invertido (10^(-2/20)) la curva conserva
    su forma pero todo el nivel cae 4.0 dB, y el dB(A) deja de ser comparable
    con un sonómetro.
    """
    if fs in _a_weight_cache:
        return _a_weight_cache[fs]

    if fs <= 0:
        raise ValueError(f"sample rate inválido: {fs}")

    # Frecuencias de polo del estándar IEC 61672-1 (Hz)
    f1, f2, f3, f4 = 20.6, 107.7, 737.9, 12194.0
    offset = 2.0  # A1000 = -2 dB

    # Numerador y denominador analógicos (ecuación E.6 del estándar)
    numerator = np.array(
        [(2.0 * np.pi * f4) ** 2.0 * (10 ** (offset / 20.0)), 0.0, 0.0, 0.0, 0.0]
    )

    part1 = [1.0, 4.0 * np.pi * f4, (2.0 * np.pi * f4) ** 2.0]
    part2 = [1.0, 4.0 * np.pi * f1, (2.0 * np.pi * f1) ** 2.0]
    part3 = [1.0, 2.0 * np.pi * f3]
    part4 = [1.0, 2.0 * np.pi * f2]
    denominator = np.convolve(np.convolve(np.convolve(part1, part2), part3), part4)

    # Transformada bilineal: analógico → digital
    b, a = bilinear(numerator, denominator, fs=fs)
    _a_weight_cache[fs] = (b, a)
    return _a_weight_cache[fs]


def apply_a_weighting(signal: np.ndarray, fs: float) -> np.ndarray:
    """Aplica filtro A-weighting al señal (float32, -1 a 1)."""
    b, a = design_a_weighting(fs)
    return lfilter(b, a, signal).astype(np.float32)


def rms_to_db(rms: float, reference: float = REFERENCE_PRESSURE) -> float:
    """
    Convierte RMS de presión (Pa) a dB SPL.
    Para audio normalizado -1..1, la referencia típica es 20e-6 Pa.

    Silencio digital (RMS 0) y valores no finitos devuelven MIN_DB en vez de
    -inf: un -inf se guarda en SQLite y sale como `-Infinity` en el JSON del
    dashboard, donde el navegador no puede parsearlo.
    """
    if not np.isfinite(rms) or rms <= 0:
        return MIN_DB
    return float(max(MIN_DB, 20.0 * np.log10(rms / reference)))


def load_config(config_path: str | Path | None = None) -> dict:
    """Carga config.yaml. Busca en cwd o ruta dada."""
    if config_path is None:
        config_path = Path.cwd() / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config no encontrado: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def resolve_path(config: dict, key: str, base_dir: Path | None = None) -> Path:
    """Resuelve ruta relativa de config.paths a absoluta."""
    base = base_dir or Path.cwd()
    raw = config["paths"][key]
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


class ConfigError(ValueError):
    """Config inválida. El mensaje nombra la sección, la clave y el motivo."""


# Secciones y claves obligatorias con su tipo esperado. Un config.yaml viejo
# (o un ejemplo desactualizado) reventaba más tarde, dentro de un thread, con
# un KeyError sin contexto: el pipeline arrancaba, el audio moría y systemd lo
# reiniciaba en bucle sin decir qué clave faltaba.
_REQUIRED_KEYS: dict[str, dict[str, tuple[type, ...]]] = {
    "audio": {
        "sample_rate": (int,),
        "block_size": (int,),
    },
    "triggers": {
        "peak_db": (int, float),
        "sustained_db": (int, float),
        "sustained_duration_ms": (int, float),
        "correlation_db": (int, float),
    },
    "buffer": {
        "audio_seconds": (int, float),
        "video_seconds": (int, float),
        "pre_event_seconds": (int, float),
        "post_event_seconds": (int, float),
    },
    "paths": {
        "database": (str,),
        "clips": (str,),
    },
    "video": {
        "resolution": (list, tuple),
        "framerate": (int,),
        "inference_interval": (int,),
        "detection_threshold": (int, float),
    },
}


def _check_number(section: str, key: str, value, minimum: float | None, maximum: float | None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{section}.{key} debe ser numérico (recibido: {value!r})")
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{section}.{key} debe ser finito (recibido: {value!r})")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{section}.{key} debe ser >= {minimum} (recibido: {value})")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{section}.{key} debe ser <= {maximum} (recibido: {value})")


def validate_config(
    config: dict,
    logger: logging.Logger | None = None,
    base_dir: Path | None = None,
) -> dict:
    """
    Comprueba que la config tiene lo que el pipeline lee, con rangos sensatos.

    Lanza ConfigError nombrando `sección.clave` en los errores duros (clave
    ausente, tipo equivocado, valor imposible) y registra avisos en los casos
    que degradan calidad sin impedir arrancar (p. ej. buffers más cortos que
    la ventana del clip, o `speed.enabled` sin las coordenadas de referencia).

    Retorna la misma config, para poder encadenar `validate_config(load_config())`.
    """
    log = logger or logger_module
    if not isinstance(config, dict):
        raise ConfigError(f"la config debe ser un diccionario (recibido: {type(config).__name__})")

    for section, keys in _REQUIRED_KEYS.items():
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"falta la sección '{section}' en la config")
        for key, types_ in keys.items():
            if key not in config[section]:
                raise ConfigError(f"falta la clave '{section}.{key}' en la config")
            value = config[section][key]
            if isinstance(value, bool) or not isinstance(value, types_):
                names = "/".join(t.__name__ for t in types_)
                raise ConfigError(
                    f"'{section}.{key}' debe ser {names} (recibido: {value!r})"
                )

    audio, triggers = config["audio"], config["triggers"]
    buffer_cfg = config["buffer"]
    video = config["video"]
    _check_number("audio", "sample_rate", audio["sample_rate"], 8000, 768000)
    _check_number("audio", "block_size", audio["block_size"], 1, None)
    _check_number("triggers", "peak_db", triggers["peak_db"], -60, 194)
    _check_number("triggers", "sustained_db", triggers["sustained_db"], -60, 194)
    _check_number("triggers", "sustained_duration_ms", triggers["sustained_duration_ms"], 0, None)
    _check_number("triggers", "correlation_db", triggers["correlation_db"], -60, 194)
    for key in ("audio_seconds", "video_seconds", "pre_event_seconds", "post_event_seconds"):
        _check_number("buffer", key, buffer_cfg[key], 0, None)

    for key in ("database", "clips"):
        if not str(config["paths"][key]).strip():
            raise ConfigError(f"'paths.{key}' no puede estar vacío")

    resolution = video["resolution"]
    if (
        len(resolution) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in resolution
        )
    ):
        raise ConfigError(
            "'video.resolution' debe ser [ancho, alto] con enteros > 0 "
            f"(recibido: {resolution!r})"
        )
    _check_number("video", "framerate", video["framerate"], 1, 240)
    _check_number("video", "inference_interval", video["inference_interval"], 1, None)
    _check_number("video", "detection_threshold", video["detection_threshold"], 0, 1)

    source = str(audio.get("source", "device"))
    if source not in ("device", "file"):
        raise ConfigError(f"'audio.source' debe ser device|file (recibido: {source!r})")
    if source == "file":
        file_path = str(audio.get("file_path") or "").strip()
        if not file_path:
            raise ConfigError("'audio.file_path' es obligatorio cuando audio.source=file")
        if base_dir is not None:
            path = Path(file_path).expanduser()
            if not path.is_absolute():
                path = Path(base_dir) / path
            if not path.is_file():
                raise ConfigError(f"'audio.file_path' no existe: {path}")

    for key, lo, hi in (
        ("calibration_offset", -200.0, 200.0),
    ):
        if key in audio:
            _check_number("audio", key, audio[key], lo, hi)
    if "cooldown_seconds" in triggers:
        _check_number("triggers", "cooldown_seconds", triggers["cooldown_seconds"], 0, None)

    classifier = config.get("audio_classifier")
    if classifier is not None and not isinstance(classifier, dict):
        raise ConfigError("'audio_classifier' debe ser una sección")
    if classifier:
        if classifier.get("backend", "cpu") not in ("cpu", "coral"):
            raise ConfigError("audio_classifier.backend debe ser cpu o coral")
        if not isinstance(classifier.get("fallback_to_cpu", True), bool):
            raise ConfigError("audio_classifier.fallback_to_cpu debe ser booleano")
        for key in ("model_path", "labels_path", "coral_model_path"):
            if key == "coral_model_path" and key not in classifier:
                continue
            value = classifier.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"audio_classifier.{key} debe ser una ruta")

    tracker = config.get("tracker")
    if tracker is not None:
        if not isinstance(tracker, dict):
            raise ConfigError("'tracker' debe ser una sección")
        for key, lo in (
            ("max_distance", 1), ("max_disappeared", 0), ("min_track_seconds", 0),
            ("min_displacement", 0), ("max_track_seconds", 0),
        ):
            if key in tracker:
                _check_number("tracker", key, tracker[key], lo, None)

    speed_cfg = config.get("speed")
    if speed_cfg is not None and not isinstance(speed_cfg, dict):
        raise ConfigError("'speed' debe ser una sección")
    if speed_cfg and not isinstance(speed_cfg.get("enabled", True), bool):
        raise ConfigError("'speed.enabled' debe ser booleano")

    if triggers["sustained_db"] > triggers["peak_db"]:
        log.warning(
            "triggers.sustained_db (%.1f) > triggers.peak_db (%.1f): el trigger "
            "sostenido nunca podrá dispararse antes que el de pico",
            triggers["sustained_db"], triggers["peak_db"],
        )

    window = buffer_cfg["pre_event_seconds"] + buffer_cfg["post_event_seconds"]
    if window > buffer_cfg["audio_seconds"]:
        log.warning(
            "buffer: pre+post = %.0fs pero audio_seconds = %.0fs; los clips "
            "empezarán después del evento",
            window, buffer_cfg["audio_seconds"],
        )
    if buffer_cfg["post_event_seconds"] > buffer_cfg["video_seconds"]:
        log.warning(
            "buffer: post_event_seconds (%.0fs) > video_seconds (%.0fs); "
            "los clips quedarán sin video al final",
            buffer_cfg["post_event_seconds"], buffer_cfg["video_seconds"],
        )

    speed = config.get("speed") or {}
    if speed.get("enabled"):
        try:
            span = abs(float(speed.get("reference_right_x", 0)) - float(speed.get("reference_left_x", 0)))
            distance = float(speed.get("reference_distance_m", 0))
        except (TypeError, ValueError):
            span = distance = 0.0
        if span <= 0 or distance <= 0:
            log.warning(
                "speed.enabled pero faltan speed.reference_left_x / reference_right_x "
                "o reference_distance_m > 0: no se estimará velocidad (ver "
                "scripts/reference_frame.py)"
            )

    return config



class UnmigratedDatabaseError(RuntimeError):
    """La base tiene el esquema anterior a la tabla única `events`."""


# Tablas del esquema anterior. Tras scripts/migrate_unified.py quedan como *_legacy.
PRE_UNIFIED_TABLES = ("noise_events", "vehicle_passages")


def _check_unified(conn: sqlite3.Connection, db_path: Path, schema_sql: str | None) -> None:
    """Falla si la base no está migrada o si `events` tiene otra forma.

    Sin esto, el código nuevo sobre una base sin migrar creaba `events` vacía y
    capturaba ahí: la migración posterior abortaba porque `events` ya tenía
    filas, y `--force` las borraba.
    """
    def exists(name: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None

    # La migración renombra las dos tablas: con el nombre original nunca es una
    # base migrada, exista o no `events` (puede ser la `events` que creó un
    # arranque en el orden equivocado).
    found = [name for name in PRE_UNIFIED_TABLES if exists(name)]
    if found:
        raise UnmigratedDatabaseError(
            f"{db_path} tiene {', '.join(found)} sin migrar: detener el servicio y "
            "correr scripts/migrate_unified.py antes de arrancar esta versión"
        )
    legacy = [f"{name}_legacy" for name in PRE_UNIFIED_TABLES if exists(f"{name}_legacy")]
    if legacy and not exists("events"):
        raise UnmigratedDatabaseError(
            f"{db_path} tiene {', '.join(legacy)} pero falta events: restaurar el backup previo"
        )
    if schema_sql is None:
        return
    reference = sqlite3.connect(":memory:")
    try:
        reference.executescript(schema_sql)
        expected = {row[1] for row in reference.execute("PRAGMA table_info(events)")}
    finally:
        reference.close()
    actual = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    if actual and expected - actual:
        raise UnmigratedDatabaseError(
            f"{db_path} tiene una tabla events con otra forma "
            f"(faltan {sorted(expected - actual)}); restaurar o migrar a mano"
        )


def init_database(config: dict | None = None, base_dir: Path | None = None) -> str:
    """
    Crea la base de datos si no existe y aplica el schema.
    Ejecuta cada sentencia por separado y tolera errores de migración.

    Antes de crear nada, se niega a seguir sobre una base con el esquema
    anterior (`noise_events`/`vehicle_passages`): hay que migrarla primero.
    Retorna la ruta a la base de datos.
    """
    if config is None:
        config = load_config()
    base = base_dir or Path.cwd()
    db_path = resolve_path(config, "database", base)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    schema_path = base / "schema.sql"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            _check_unified(conn, db_path, None)
        except UnmigratedDatabaseError:
            conn.close()
            raise
        # WAL: un lector largo (export CSV en streaming) no bloquea a los
        # writers de captura/eventos. Es persistente para esta base.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error as exc:
            logger_module.warning("No se pudo activar WAL en %s: %s", db_path, exc)
        for statement in schema_sql.split(";"):
            # Strip comments and whitespace to check if there's real SQL
            lines = [
                l for l in statement.strip().splitlines()
                if l.strip() and not l.strip().startswith("--")
            ]
            if not lines:
                continue
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass  # tolerate migration errors
        conn.commit()
        try:
            _check_unified(conn, db_path, schema_sql)
        finally:
            conn.close()

    return str(db_path)


def get_connection(db_path: str) -> sqlite3.Connection:
    """Obtiene conexión SQLite (para uso en threads, cada uno su conexión)."""
    return sqlite3.connect(db_path, check_same_thread=False)
