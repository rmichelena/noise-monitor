"""Análisis de clips fuera de la ruta de captura."""

import hashlib
import json
import math
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import freqz, lfilter, resample_poly

from .analysis_store import ROAD_VEHICLES
from .audio_classifier import AudioClassifier, YAMNET_EXPECTED_SAMPLES, YAMNET_SAMPLE_RATE
from .utils import REFERENCE_PRESSURE, design_a_weighting

ALGORITHM_VERSION = "2"
ANNOYANCE_MODEL = "Widmann 1992 / Fastl–Zwicker 16.2–16.4"
ANNOYANCE_MIN_SECONDS = 2
MAX_CLIP_SECONDS = 60
PSYCHOACOUSTIC_RATE = 48000
YAMNET_HOP = 7680
VEHICLE_LABELS = {
    "car": {"Car", "Car passing by"},
    "motorcycle": {"Motorcycle"},
    "truck": {"Truck"},
    "bus": {"Bus"},
}
ROAD_LABELS = set().union(*VEHICLE_LABELS.values(), {
    "Vehicle", "Motor vehicle (road)", "Vehicle horn, car horn, honking",
    "Air horn, truck horn", "Engine", "Accelerating, revving, vroom",
    "Traffic noise, roadway noise",
})
INDOOR_LABELS = {"Dishes, pots, and pans", "Cutlery, silverware", "Television"}
NON_ROAD_COMPETITORS = INDOOR_LABELS | {"Chainsaw", "Lawn mower", "Power tool", "Blender", "Microwave oven"}
BEHAVIOR_LABELS = {
    "vehicular_horn": {"Vehicle horn, car horn, honking", "Air horn, truck horn"},
    "acceleration_revving": {"Accelerating, revving, vroom"},
    "engine_start": {"Engine starting"},
    "idling": {"Idling"},
    "dish_clatter": {"Dishes, pots, and pans", "Cutlery, silverware"},
}


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def validate_calibration(spec: dict | None) -> dict | None:
    if spec is None:
        return None
    if not isinstance(spec, dict) or not str(spec.get("id") or "").strip():
        raise ValueError("la calibración requiere id")
    factor = float(spec.get("pa_per_unit", math.nan))
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("pa_per_unit debe ser positivo y finito")
    return {**spec, "pa_per_unit": factor}


def read_clip(path: Path) -> tuple[np.ndarray, int, str, float]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,codec_name,channels,bits_per_raw_sample,bits_per_sample", "-of", "json", str(path)],
        check=True, capture_output=True, timeout=20,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams:
        raise ValueError("clip sin pista de audio")
    stream = streams[0]
    if stream["channels"] != 1:
        raise ValueError("se requiere una pista mono del micrófono")
    sample_rate = int(stream["sample_rate"])
    if not 8000 <= sample_rate <= 192000:
        raise ValueError("frecuencia de muestreo fuera de rango")
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
         "-map", "0:a:0", "-t", str(MAX_CLIP_SECONDS + .1), "-f", "f32le", "pipe:1"],
        check=True, capture_output=True, timeout=45,
    )
    samples = np.frombuffer(decode.stdout, dtype="<f4")
    if len(samples) > sample_rate * MAX_CLIP_SECONDS:
        raise ValueError(f"clip supera {MAX_CLIP_SECONDS} s")
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("audio vacío o con muestras inválidas")
    bits = stream.get("bits_per_raw_sample") or stream.get("bits_per_sample")
    integer_pcm_or_lossless = stream["codec_name"].startswith(("pcm_s", "pcm_u")) or stream["codec_name"] == "flac"
    digital_limit = 1.0
    if integer_pcm_or_lossless and str(bits).isdigit() and 8 <= int(bits) <= 32:
        digital_limit -= 2.0 ** (1 - int(bits))
    return samples, sample_rate, stream["codec_name"], digital_limit


def resample(samples: np.ndarray, source: int, target: int) -> np.ndarray:
    divisor = math.gcd(source, target)
    return resample_poly(samples, target // divisor, source // divisor)


def infer_source(
    labels: list[dict], visual_labels: list[str], visual_vehicle_confirmed: bool = False,
) -> dict:
    """Hipótesis espacial basada en evidencia concordante y sin conflictos."""
    active = {label["label"] for label in labels}
    road_audio = ROAD_LABELS.intersection(active)
    indoor = INDOOR_LABELS.intersection(active)
    road_visual = set(ROAD_VEHICLES).intersection(visual_labels)
    audio_vehicles = {kind for kind, names in VEHICLE_LABELS.items() if names.intersection(active)}
    source = {
        "source_attribution": "uncertain", "vehicle_source": None, "confidence": None,
        "acoustic_behaviors": [behavior for behavior, names in BEHAVIOR_LABELS.items()
                               if names.intersection(active)],
    }
    if indoor and not road_audio and not road_visual:
        source["source_attribution"] = "indoor"
    elif (
        visual_vehicle_confirmed and not NON_ROAD_COMPETITORS.intersection(active)
        and len(road_visual) == 1 and road_visual == audio_vehicles
    ):
        source.update(source_attribution="outdoor", vehicle_source=next(iter(road_visual)))
    return source


class ClipAnalyzer:
    def __init__(
        self, model: Path, labels: Path, min_score: float = .3, field: str = "free",
        *, annoyance: bool = False,
    ):
        from mosqito import sq_metrics

        if not 0 < min_score <= 1 or field not in {"free", "diffuse"}:
            raise ValueError("min_score o campo acústico inválido")
        if annoyance and field != "free":
            raise ValueError("el modelo de fluctuación para PA requiere campo libre")
        self.annoyance_metrics = None
        if annoyance:
            try:
                from phonometry import psychoacoustics
            except ImportError as exc:
                raise RuntimeError("PA requiere instalar requirements-analysis.txt") from exc
            self.annoyance_metrics = psychoacoustics
        self.metrics = sq_metrics
        self.classifier = AudioClassifier(model, labels, {})
        if len(self.classifier.labels) != 521:
            raise ValueError("se requieren las 521 etiquetas de YAMNet")
        decoder = subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()[0]
        self.parameters = {
            "ffmpeg_version": decoder.split(" version ", 1)[1].split()[0],
            "algorithm_version": ALGORITHM_VERSION,
            "mosqito_version": version("mosqito"),
            "litert_version": version("ai-edge-litert"),
            "numpy_version": version("numpy"),
            "scipy_version": version("scipy"),
            "model_sha256": file_sha256(model),
            "labels_sha256": file_sha256(labels),
            "min_score": min_score,
            "yamnet_window_samples": YAMNET_EXPECTED_SAMPLES,
            "yamnet_hop_samples": YAMNET_HOP,
            "yamnet_sample_rate_hz": YAMNET_SAMPLE_RATE,
            "yamnet_aggregation": "max_across_windows, includes final full window or left-padded short clip",
            "yamnet_source_ranking": "max score by class, Silence recorded separately",
            "field": field,
            "psychoacoustic_sample_rate_hz": PSYCHOACOUSTIC_RATE,
            "sharpness": "DIN 45692, mean where loudness > 0",
            "roughness": "Daniel and Weber, mean, 50% overlap",
            "tonality": "ECMA tone-to-noise ratio in dB, maximum across 0.5 s segments",
            "lafmax_time_constant_s": .125,
            "a_weighting_reference_hz": 1000,
        }
        if annoyance:
            self.parameters["annoyance"] = {
                "model": ANNOYANCE_MODEL,
                "phonometry_version": version("phonometry"),
                "min_duration_s": ANNOYANCE_MIN_SECONDS,
                "fluctuation_strength": "Osses et al. 2016 (phonometry), median across 2 s frames, 50% overlap",
                "sharpness_and_roughness": "same clip means as the separate MoSQITo metrics",
            }

    def classify_clip(self, samples: np.ndarray, rate: int) -> dict:
        waveform = resample(np.clip(samples, -1, 1), rate, YAMNET_SAMPLE_RATE)
        last = max(0, len(waveform) - YAMNET_EXPECTED_SAMPLES)
        offsets = list(range(0, last, YAMNET_HOP)) + [last]
        scores = np.maximum.reduce([
            self.classifier.predict_scores(
                waveform[start:start + YAMNET_EXPECTED_SAMPLES], YAMNET_SAMPLE_RATE
            )
            for start in offsets
        ])
        if not np.isfinite(scores).all():
            raise ValueError("YAMNet devolvió scores inválidos")
        order = np.argsort(-scores, kind="stable")
        labels = [
            {"label": self.classifier.labels[idx], "score": float(scores[idx])}
            for idx in order
            if scores[idx] >= self.parameters["min_score"]
        ]
        return {
            "audio_labels": [label for label in labels if label["label"] != "Silence"],
            "silence_score": next((label["score"] for label in labels if label["label"] == "Silence"), None),
            "raw_yamnet_top_label": self.classifier.labels[order[0]],
        }

    def analyze(
        self, path: Path, *, calibration: dict | None = None,
        visual_labels: list[str] | None = None, human_review: dict | None = None,
        audio_muted: bool = False, visual_vehicle_confirmed: bool = False,
    ) -> dict:
        start = perf_counter()
        visual_labels = sorted(set(visual_labels or []))
        calibration_issue = None
        try:
            calibration = validate_calibration(calibration)
        except (TypeError, ValueError) as exc:
            calibration_issue = str(exc)
            calibration = None
        result = {
            "clip_path": str(path), "clip_sha256": None, "duration_s": None,
            "audio": {"laeq_dba": None, "lafmax_dba": None, "calibration_id": None,
                      "sample_rate_hz": None, "codec": None},
            "psychoacoustics": {
                "loudness_n5_sone": None, "sharpness_acum": None, "roughness_asper": None,
                "tonality": None, "fluctuation_strength_vacil": None, "model_annoyance_score": None,
                "algorithm_version": ALGORITHM_VERSION,
            },
            "classification": {
                "audio_labels": [], "audio_model": "yamnet", "visual_labels": visual_labels,
                "silence_score": None, "raw_yamnet_top_label": None,
                "visual_vehicle_confirmed": visual_vehicle_confirmed,
                **infer_source([], visual_labels),
            },
            "human_review": {"perceived_loudness": None, "annoyance_rating": None,
                             **(human_review or {})},
            "parameters": self.parameters, "calibration": calibration,
            "issues": {}, "status": "completed",
        }
        issues = result["issues"]
        try:
            if audio_muted:
                raise ValueError("audio retirado por privacidad")
            result["clip_sha256"] = file_sha256(path)
            samples, rate, codec, digital_limit = read_clip(path)
            if file_sha256(path) != result["clip_sha256"]:
                raise ValueError("el clip cambió durante la lectura")
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            issues["audio"] = str(exc)
            result["status"] = "unavailable" if audio_muted else "failed"
        else:
            result["duration_s"] = len(samples) / rate
            result["audio"].update(sample_rate_hz=rate, codec=codec)
            try:
                classification = self.classify_clip(samples, rate)
                result["classification"].update(
                    **classification,
                    **infer_source(classification["audio_labels"], visual_labels, visual_vehicle_confirmed),
                )
            except (ValueError, RuntimeError) as exc:
                issues["yamnet"] = str(exc)
                result["status"] = "failed"

            if calibration is None:
                issues["calibration"] = calibration_issue or "sin calibración absoluta para este clip"
            elif calibration.get("codec", codec) != codec:
                issues["calibration"] = "calibración incompatible con el códec de la pista"
            elif np.max(np.abs(samples)) >= digital_limit:
                issues["calibration"] = "audio saturado o con sobremodulación"
            else:
                result["audio"]["calibration_id"] = calibration["id"]
                self._physical_metrics(samples.astype(np.float64) * calibration["pa_per_unit"],
                                       rate, result)
        result["elapsed_ms"] = round((perf_counter() - start) * 1000, 1)
        return result

    def _physical_metrics(self, pressure: np.ndarray, rate: int, result: dict) -> None:
        audio, psycho, issues = (result[key] for key in ("audio", "psychoacoustics", "issues"))
        b, a = design_a_weighting(rate)
        # La normalización explícita también reproduce los resultados tras corregir
        # la ganancia del filtro legado en el pipeline de captura.
        gain = abs(freqz(b, a, worN=[1000.0], fs=rate)[1][0])
        energy = lfilter(b / gain, a, pressure) ** 2
        mean = float(energy.mean())
        if math.isfinite(mean) and mean > 0:
            audio["laeq_dba"] = 10 * math.log10(mean / REFERENCE_PRESSURE ** 2)
            if len(energy) >= .125 * rate:
                alpha = math.exp(-1 / (.125 * rate))
                fast = float(lfilter([1 - alpha], [1, -alpha], energy).max())
                if fast > 0:
                    audio["lafmax_dba"] = 10 * math.log10(fast / REFERENCE_PRESSURE ** 2)
        else:
            issues["spl"] = "sin potencia acústica medible"

        if len(pressure) < math.ceil(rate / 2):
            issues["psychoacoustics"] = "la comparación requiere un clip de al menos 0,5 s"
            return

        waveform = resample(pressure, rate, PSYCHOACOUSTIC_RATE)
        try:
            loudness, specific, *_ = self.metrics.loudness_zwtv(
                waveform, PSYCHOACOUSTIC_RATE, field_type=self.parameters["field"],
            )
            psycho["loudness_n5_sone"] = finite(np.percentile(loudness, 95))
            if np.any(loudness > 0):
                sharpness = self.metrics.sharpness_din_from_loudness(loudness, specific)
                psycho["sharpness_acum"] = finite(np.mean(sharpness[loudness > 0]))
        except (ArithmeticError, ValueError, IndexError, RuntimeError) as exc:
            issues["loudness_sharpness"] = str(exc)
        try:
            roughness, *_ = self.metrics.roughness_dw(waveform, PSYCHOACOUSTIC_RATE)
            psycho["roughness_asper"] = finite(np.mean(roughness))
        except (ArithmeticError, ValueError, IndexError, RuntimeError) as exc:
            issues["roughness"] = str(exc)
        if np.any(waveform):
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    size = PSYCHOACOUSTIC_RATE // 2
                    ratios = [
                        self.metrics.tnr_ecma_st(patch, PSYCHOACOUSTIC_RATE)[0]
                        for start in range(0, len(waveform) - size + 1, size)
                        if np.any(patch := waveform[start:start + size])
                    ]
                if ratios:
                    psycho["tonality"] = finite(np.max(ratios))
            except (ArithmeticError, ValueError, IndexError, RuntimeError) as exc:
                issues["tonality"] = str(exc)
        if self.annoyance_metrics is not None:
            self._experimental_annoyance(waveform, psycho, issues)

    def _experimental_annoyance(self, waveform: np.ndarray, psycho: dict, issues: dict) -> None:
        if len(waveform) < ANNOYANCE_MIN_SECONDS * PSYCHOACOUSTIC_RATE:
            issues["annoyance"] = "PA requiere al menos 2 s de audio calibrado"
            return
        try:
            fluctuation = self.annoyance_metrics.fluctuation_strength(waveform, PSYCHOACOUSTIC_RATE)
            strength = finite(fluctuation.fluctuation_strength)
            psycho["fluctuation_strength_vacil"] = strength
            n5, sharpness, roughness = (psycho[key] for key in (
                "loudness_n5_sone", "sharpness_acum", "roughness_asper",
            ))
            if None in (n5, sharpness, strength, roughness):
                issues["annoyance"] = "falta una de las cuatro métricas del modelo PA"
                return
            score = self.annoyance_metrics.psychoacoustic_annoyance(n5, sharpness, strength, roughness)
            psycho["model_annoyance_score"] = finite(score.annoyance)
        except (ArithmeticError, ValueError, IndexError, RuntimeError) as exc:
            issues["annoyance"] = str(exc)


def finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None
