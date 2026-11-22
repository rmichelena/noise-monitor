"""Score rolling, calibrated microphone windows in a separate process."""

import logging
import math
import multiprocessing
import threading
import time
from collections import deque
from dataclasses import dataclass
from importlib.metadata import version
from queue import Empty, Full, Queue

import numpy as np

logger = logging.getLogger(__name__)
METRICS = {
    "loudness_mean": ("ISO 532-1 Zwicker time-varying mean", "sone"),
    "annoyance_pa": ("Widmann PA (ISO 532-1 / DIN / ECMA roughness / Osses)", "PA"),
}


@dataclass(frozen=True)
class Settings:
    metric: str
    threshold: float
    pa_per_unit: float
    calibration_id: str
    window_seconds: float
    hop_seconds: float
    result_age_limit: float

    @classmethod
    def load(cls, config: dict):
        triggers = config["triggers"]
        criterion = triggers.get("criterion", "dba")
        if criterion == "dba":
            return None
        if criterion != "psychoacoustic":
            raise ValueError("triggers.criterion debe ser dba o psychoacoustic")
        values = triggers.get("psychoacoustic") or {}
        calibration = (config["audio"].get("pressure_calibration") or {})
        metric = values.get("metric", "loudness_mean")
        if metric not in METRICS:
            raise ValueError("triggers.psychoacoustic.metric debe ser loudness_mean o annoyance_pa")
        calibration_id = calibration.get("id")
        if not isinstance(calibration_id, str) or not calibration_id.strip():
            raise ValueError("el criterio psicoacústico requiere audio.pressure_calibration.id")

        def positive(value, name):
            if isinstance(value, bool):
                raise ValueError(f"{name} debe ser positivo y finito")
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name} debe ser positivo y finito") from None
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} debe ser positivo y finito")
            return value

        threshold = positive(values.get("threshold"), "triggers.psychoacoustic.threshold")
        factor = positive(calibration.get("pa_per_unit"), "audio.pressure_calibration.pa_per_unit")
        window = positive(values.get("window_seconds", 2 if metric == "annoyance_pa" else 3), "window_seconds")
        hop = positive(values.get("hop_seconds", 2 if metric == "annoyance_pa" else 1.5), "hop_seconds")
        buffer = config["buffer"]
        pre, post = int(buffer["pre_event_seconds"]), int(buffer["post_event_seconds"])
        if window < 2 or window > 2 * min(pre, post) or hop > window:
            raise ValueError("la ventana debe tener al menos 2 s, caber en el clip y cubrir cada salto")
        buffer_seconds = min(int(buffer["audio_seconds"]), int(buffer["video_seconds"]))
        if buffer_seconds < pre + post:
            raise ValueError("los buffers de audio y video deben cubrir todo el clip")
        # Reserve one audio block for timestamp quantization at the ring boundary.
        block_time = config["audio"]["block_size"] / config["audio"]["sample_rate"]
        max_age = buffer_seconds - pre - block_time
        # La ventana se centra medio window antes del último bloque; el
        # resultado llega tras el hop y la latencia del scorer. Todo eso debe
        # caber antes de que el pre-roll se evacue.
        default_latency = 5.6 if metric == "annoyance_pa" else 1.6
        expected_latency = positive(
            values.get("expected_latency_seconds", default_latency),
            "triggers.psychoacoustic.expected_latency_seconds",
        )
        required = window / 2 + block_time + hop + expected_latency
        if max_age < required:
            raise ValueError(
                f"{metric} requiere {required:.1f} s de margen "
                f"(audio/video >= {pre + required + block_time:.1f} s con pre={pre}, "
                f"hop={hop}, latencia esperada={expected_latency}); "
                f"configuración actual: {max_age:.1f} s"
            )
        return cls(metric, threshold, factor, calibration_id, window, hop, max_age)


def score_window(samples: np.ndarray, sample_rate: int, settings: Settings, metrics) -> float | None:
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) >= 1:
        return None
    pressure = samples.astype(np.float64) * settings.pa_per_unit
    if settings.metric == "loudness_mean":
        loudness = metrics.loudness_zwicker(pressure, sample_rate, field="free")
        score = float(np.mean(loudness.loudness_vs_time))
    else:
        score = float(metrics.psychoacoustic_annoyance_from_signal(pressure, sample_rate, field="free").annoyance)
    return score if math.isfinite(score) and score >= 0 else None


def _score_process(connection, sample_rate: int, settings: Settings):
    try:
        try:
            from phonometry import psychoacoustics
            connection.send((version("phonometry"), None))
        except ImportError:
            connection.send((None, "instalá requirements-trigger.txt en el entorno de captura"))
            return
        while True:
            samples = connection.recv()
            try:
                connection.send((score_window(samples, sample_rate, settings, psychoacoustics), None))
            except (ValueError, RuntimeError, ArithmeticError, IndexError) as exc:
                connection.send((None, str(exc)))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class PerceptualTrigger:
    def __init__(self, settings: Settings, sample_rate: int, block_size: int, emit):
        self.settings = settings
        self.sample_rate = sample_rate
        self.emit = emit
        self.window_samples = round(sample_rate * settings.window_seconds)
        self.window_blocks = math.ceil(self.window_samples / block_size)
        self.hop_blocks = max(1, round(sample_rate * settings.hop_seconds / block_size))
        self._blocks = deque(maxlen=self.window_blocks)
        self._next_window = self.window_blocks
        self._pending = Queue(maxsize=1)
        self._stop = threading.Event()
        self._process = self._thread = self._connection = None
        self.model_version = None
        self.evaluated = self.dropped = self.expired = self.invalid = 0
        self.event_queue_full = 0
        self.restarts = 0
        self.last_error = None
        self.last_score = self.last_latency_ms = None
        self._in_flight_deadline = None

    def _kill_worker(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
        if self._connection is not None:
            try:
                self._connection.close()
            except OSError:
                pass
        self._process = self._connection = None

    def _spawn_worker(self):
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        process = ctx.Process(target=_score_process, args=(child, self.sample_rate, self.settings),
                              name="psychoacoustic-scorer", daemon=True)
        process.start()
        child.close()
        self._process, self._connection = process, parent
        try:
            if not parent.poll(20):
                raise RuntimeError("el proceso de disparo psicoacústico no inició a tiempo")
            self.model_version, error = parent.recv()
            if error:
                raise RuntimeError(error)
        except (EOFError, RuntimeError):
            self._kill_worker()
            raise

    def start(self):
        self._stop.clear()
        self._spawn_worker()
        self._thread = threading.Thread(target=self._dispatch, name="psychoacoustic-trigger", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._kill_worker()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def offer(self, timestamp: float, samples: np.ndarray, db: float):
        """Capture callback merely hands off references to a recent PCM window."""
        if self._blocks and timestamp - self._blocks[-1][0] > 2.5 * len(samples) / self.sample_rate:
            self._blocks.clear()
            self._next_window = self.window_blocks
        self._blocks.append((timestamp, samples, db))
        self._next_window -= 1
        if self._next_window > 0:
            return
        self._next_window = self.hop_blocks
        if len(self._blocks) < self.window_blocks:
            return
        item = tuple(self._blocks)
        try:
            self._pending.put_nowait(item)
        except Full:
            # Prefer current audio over scheduling old work after saturation.
            try:
                self._pending.get_nowait()
                self.dropped += 1
            except Empty:
                pass
            self._pending.put_nowait(item)

    def status(self):
        model, unit = METRICS[self.settings.metric]
        thread, process, deadline = self._thread, self._process, self._in_flight_deadline
        return {
            "criterion": "psychoacoustic", "metric": self.settings.metric, "model": model,
            "unit": unit, "threshold": self.settings.threshold, "last_score": self.last_score,
            "last_latency_ms": self.last_latency_ms, "evaluated": self.evaluated,
            "dropped": self.dropped, "expired": self.expired, "invalid": self.invalid,
            "event_queue_full": self.event_queue_full,
            "restarts": self.restarts, "last_error": self.last_error,
            "running": bool(thread and thread.is_alive() and process and process.is_alive()),
            "overdue": deadline is not None and time.time() >= deadline,
        }

    def _respawn_worker(self):
        delay = 2.0
        while not self._stop.is_set():
            if self._stop.wait(delay):
                return None
            try:
                self._spawn_worker()
                self.restarts += 1
                logger.warning("Scorer psicoacústico reiniciado (intento %d)", self.restarts)
                return self._connection
            except Exception as exc:
                self.last_error = str(exc)
                logger.error("No se pudo reiniciar el scorer psicoacústico: %s", exc)
                delay = min(delay * 2, 60)
        return None

    def _dispatch(self):
        conn = self._connection
        while not self._stop.is_set():
            if conn is None:
                conn = self._respawn_worker()
                if conn is None:
                    return
            try:
                window = self._pending.get(timeout=.2)
            except Empty:
                continue
            timestamp = window[-1][0] - self.window_samples / (2 * self.sample_rate)
            if time.time() - timestamp >= self.settings.result_age_limit:
                self.expired += 1
                continue
            samples = np.concatenate([block for _, block, _ in window])[-self.window_samples:]
            self._in_flight_deadline = timestamp + self.settings.result_age_limit
            timed_out = False
            try:
                conn.send(samples)
                while not self._stop.is_set() and not conn.poll(.2):
                    if (
                        self._in_flight_deadline is not None
                        and time.time() >= self._in_flight_deadline
                    ):
                        timed_out = True
                        break
                if self._stop.is_set():
                    return
                if timed_out:
                    raise TimeoutError("el scorer no respondió antes del deadline")
                score, error = conn.recv()
            except (EOFError, OSError, TimeoutError) as exc:
                self.last_error = str(exc)
                logger.error("Se detuvo el scorer psicoacústico: %s", exc)
                self.dropped += 1
                self._kill_worker()
                conn = None
                continue
            finally:
                self._in_flight_deadline = None
            self.last_error = None
            self.last_latency_ms = round((time.time() - window[-1][0]) * 1000)
            if error or score is None:
                self.invalid += 1
                if error and self.invalid == 1:
                    logger.warning("No se pudo evaluar una ventana psicoacústica: %s", error)
                continue
            self.last_score = score
            self.evaluated += 1
            if time.time() - timestamp >= self.settings.result_age_limit:
                self.expired += 1
            elif score >= self.settings.threshold:
                self._emit(timestamp, score, window, samples)

    def _emit(self, timestamp, score, window, samples):
        peak = max(db for _, _, db in window)
        if not math.isfinite(peak):
            return
        mean_db = peak + 10 * math.log10(sum(10 ** ((db - peak) / 10) for _, _, db in window) / len(window))
        model, unit = METRICS[self.settings.metric]
        event = {
            "type": "psychoacoustic", "timestamp": timestamp, "db": peak, "db_avg": mean_db,
            "expires_at": timestamp + self.settings.result_age_limit,
            "trigger_decision": {
                "metric": self.settings.metric, "unit": unit, "model": model,
                "phonometry_version": self.model_version, "score": score, "threshold": self.settings.threshold,
                "calibration_id": self.settings.calibration_id, "window_s": len(samples) / self.sample_rate,
            },
        }
        try:
            self.emit(event)
        except Full:
            self.event_queue_full += 1
