"""
Utilidades base para el sistema de monitoreo de ruido.
- Filtro A-weighting IEC 61672-1
- Conversión RMS → dB
- Carga de configuración
- Inicialización de SQLite
"""

import os
import sqlite3
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import bilinear, lfilter

# Presión de referencia estándar (20 µPa)
REFERENCE_PRESSURE = 20e-6
SPEECH_CLASSES = {
    "Speech", "Child speech", "Child speech, kid speaking", "Conversation",
    "Narration", "Narration, monologue",
    "Babbling", "Speech synthesizer", "Whispering",
}

# Coeficientes del filtro A-weighting (precalculados por sample rate)
_a_weight_cache = {}


def design_a_weighting(fs: float):
    """
    Diseña filtro A-weighting según IEC 61672-1.
    Retorna coeficientes (b, a) para lfilter (digital).
    """
    if fs in _a_weight_cache:
        return _a_weight_cache[fs]

    # Frecuencias de polo del estándar IEC 61672-1 (Hz)
    f1, f2, f3, f4 = 20.6, 107.7, 737.9, 12194.0
    offset = 2.0  # A1000 = -2 dB

    # Numerador y denominador analógicos (ecuación E.6 del estándar)
    numerator = np.array(
        [(2.0 * np.pi * f4) ** 2.0 * (10 ** (-offset / 20.0)), 0.0, 0.0, 0.0, 0.0]
    )
    part1 = [1.0, 4.0 * np.pi * f4, (2.0 * np.pi * f4) ** 2.0]
    part2 = [1.0, 4.0 * np.pi * f1, (2.0 * np.pi * f1) ** 2.0]
    part3 = [1.0, 2.0 * np.pi * f3]
    part4 = [1.0, 2.0 * np.pi * f2]
    denominator = np.convolve(np.convolve(np.convolve(part1, part2), part3), part4)

    # Transformada bilinear: analógico → digital
    b, a = bilinear(numerator, denominator, fs=fs)
    _a_weight_cache[fs] = (b, a)
    return b, a


def apply_a_weighting(signal: np.ndarray, fs: float) -> np.ndarray:
    """Aplica filtro A-weighting al señal (float32, -1 a 1)."""
    b, a = design_a_weighting(fs)
    return lfilter(b, a, signal).astype(np.float32)


def rms_to_db(rms: float, reference: float = REFERENCE_PRESSURE) -> float:
    """
    Convierte RMS de presión (Pa) a dB SPL.
    Para audio normalizado -1..1, la referencia típica es 20e-6 Pa.
    """
    if rms <= 0:
        return -np.inf
    return 20.0 * np.log10(rms / reference)


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


def init_database(config: dict | None = None, base_dir: Path | None = None) -> str:
    """
    Crea la base de datos si no existe y aplica el schema.
    Ejecuta each statement individually to tolerate migration errors
    (e.g. source table doesn't exist in fresh DBs).
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

        conn = sqlite3.connect(str(db_path))
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
        conn.close()

    return str(db_path)


def get_connection(db_path: str) -> sqlite3.Connection:
    """Obtiene conexión SQLite (para uso en threads, cada uno su conexión)."""
    return sqlite3.connect(db_path, check_same_thread=False)
