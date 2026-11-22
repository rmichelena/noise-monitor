"""
Clasificación de sonido ambiental con YAMNet (521 clases AudioSet, TFLite).
Corre en CPU vía ai-edge-litert. Resamplea de 44.1 kHz a 16 kHz con resampy.
"""

import logging
import threading
from pathlib import Path

import numpy as np
import resampy
from ai_edge_litert.interpreter import Interpreter

from .utils import SPEECH_CLASSES

logger = logging.getLogger(__name__)

YAMNET_SAMPLE_RATE = 16000
# YAMNet expects exactly 0.975s of audio at 16kHz
YAMNET_EXPECTED_SAMPLES = 15600


class AudioClassifier:
    """
    Clasifica ventanas de audio usando YAMNet.
    Input:  audio float32 a cualquier sample rate.
    Output: (sound_class, sound_category, score).
    """

    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        category_map: dict[str, list[str]],
        min_score: float = 0.3,
    ):
        self.min_score = min_score
        self.category_map = category_map

        self._lock = threading.Lock()
        self._interpreter = Interpreter(model_path=str(model_path))

        # Pre-allocate with the exact expected input size
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        self._input_index = input_details[0]["index"]
        self._output_index = output_details[0]["index"]
        self._interpreter.resize_tensor_input(
            self._input_index, [YAMNET_EXPECTED_SAMPLES]
        )
        self._interpreter.allocate_tensors()

        self.labels = self._load_labels(labels_path)
        self._class_to_category: dict[str, str] = {}
        self._build_reverse_map()

        logger.info(
            "AudioClassifier cargado: %s (%d labels, %d categorías)",
            Path(model_path).name,
            len(self.labels),
            len(category_map),
        )

    @staticmethod
    def _load_labels(path: str | Path) -> list[str]:
        path = Path(path)
        if not path.exists():
            logger.warning("Labels no encontrado: %s", path)
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _build_reverse_map(self):
        self._class_to_category = {}
        for category, class_names in self.category_map.items():
            for name in class_names:
                self._class_to_category[name.lower()] = category

    def _lookup_category(self, class_name: str) -> str:
        lower = class_name.lower()
        for label_fragment, cat in self._class_to_category.items():
            if label_fragment in lower:
                return cat
        return "other"

    def predict_scores(
        self, audio_window: np.ndarray, source_sr: int = 44100
    ) -> np.ndarray:
        """Obtiene los scores AudioSet de una ventana de 0,975 s."""
        if source_sr != YAMNET_SAMPLE_RATE:
            audio = resampy.resample(
                audio_window.astype(np.float32), source_sr, YAMNET_SAMPLE_RATE
            )
        else:
            audio = audio_window.astype(np.float32)

        # YAMNet requires exactly 15600 samples — pad or truncate
        if len(audio) > YAMNET_EXPECTED_SAMPLES:
            audio = audio[-YAMNET_EXPECTED_SAMPLES:]
        elif len(audio) < YAMNET_EXPECTED_SAMPLES:
            audio = np.pad(audio, (YAMNET_EXPECTED_SAMPLES - len(audio), 0))

        with self._lock:
            self._interpreter.set_tensor(self._input_index, audio)
            self._interpreter.invoke()
            scores = self._interpreter.get_tensor(self._output_index)
        # scores shape: (patches, 521) — average across patches
        if scores.ndim > 1:
            scores = scores.mean(axis=0)

        return scores

    def classify(
        self, audio_window: np.ndarray, source_sr: int = 44100
    ) -> tuple[str, str, float]:
        """Mantiene la clasificación de etiqueta única usada por la captura."""
        return self._best_label(self.predict_scores(audio_window, source_sr))

    def classify_with_speech(self, audio_window: np.ndarray, source_sr: int):
        """Detecta habla aunque otra etiqueta tenga el mayor score."""
        if not np.isfinite(audio_window).all():
            raise ValueError("audio no finito")
        scores = self.predict_scores(audio_window, source_sr)
        if scores.shape != (len(self.labels),) or not np.isfinite(scores).all():
            raise ValueError("scores o mapa de etiquetas YAMNet inválidos")
        speech = any(label in SPEECH_CLASSES and score >= self.min_score
                     for label, score in zip(self.labels, scores))
        return (*self._best_label(scores), speech)

    def _best_label(self, scores):
        top_idx = int(np.argmax(scores))
        top_score = float(scores[top_idx])

        if top_idx >= len(self.labels):
            return ("unknown", "other", top_score)

        sound_class = self.labels[top_idx]

        if top_score < self.min_score:
            return ("unknown", "other", top_score)

        category = self._lookup_category(sound_class)
        return (sound_class, category, top_score)
