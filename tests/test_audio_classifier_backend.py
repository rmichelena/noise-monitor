import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.audio_classifier import AudioClassifier, YAMNET_EXPECTED_SAMPLES
from src.utils import ConfigError, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


def interpreter():
    runtime = Mock()
    runtime.get_input_details.return_value = [
        {"index": 2, "shape": [YAMNET_EXPECTED_SAMPLES], "dtype": np.float32},
    ]
    runtime.get_output_details.return_value = [{"index": 8}]
    runtime.get_tensor.return_value = np.zeros((1, 521), dtype=np.float32)
    return runtime


class AudioBackendTests(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(tmp)
        self.cpu = root / "yamnet.tflite"
        self.cpu.write_bytes(b"CPU")
        self.coral = root / "yamnet_edgetpu.tflite"
        self.coral.write_bytes(b"edgetpu-custom-op")
        self.labels = root / "labels.txt"
        self.labels.write_text("".join(f"Label {index}\n" for index in range(521)))
        self.make_interpreter = self.enterContext(patch("src.audio_classifier.Interpreter"))
        self.load_delegate = self.enterContext(patch("src.audio_classifier.load_delegate"))

    def classifier(self, **kwargs):
        return AudioClassifier(self.cpu, self.labels, {}, **kwargs)

    def test_cpu_default_does_not_load_coral(self):
        self.make_interpreter.return_value = interpreter()
        classifier = self.classifier()
        self.make_interpreter.assert_called_once_with(model_path=str(self.cpu))
        self.load_delegate.assert_not_called()
        self.assertEqual(classifier.status()["active_backend"], "cpu")

    def test_coral_warmup_uses_fixed_waveform_and_compiled_model(self):
        runtime = interpreter()
        self.make_interpreter.return_value = runtime
        classifier = self.classifier(backend="coral")
        self.load_delegate.assert_called_once_with("libedgetpu.so.1", options={"device": "usb"})
        self.make_interpreter.assert_called_once_with(
            model_path=str(self.coral), experimental_delegates=[self.load_delegate.return_value],
        )
        runtime.resize_tensor_input.assert_not_called()
        runtime.invoke.assert_called_once()
        self.assertEqual(classifier.status()["active_backend"], "coral")
        self.assertEqual(classifier.status()["inference_count"], 0)

    def test_strict_coral_rejects_cpu_artifact(self):
        with self.assertRaisesRegex(ValueError, "operador compilado"):
            self.classifier(backend="coral", coral_model_path=self.cpu)
        self.load_delegate.assert_not_called()

    def test_startup_failure_can_fall_back_and_reports_actual_backend(self):
        self.load_delegate.side_effect = RuntimeError("USB ausente")
        self.make_interpreter.return_value = interpreter()
        classifier = self.classifier(backend="coral", fallback_to_cpu=True)
        self.assertEqual(classifier.status()["requested_backend"], "coral")
        self.assertEqual(classifier.status()["active_backend"], "cpu")
        self.assertEqual(classifier.status()["fallbacks"], 1)

    def test_runtime_failure_retries_once_on_cpu(self):
        coral, cpu = interpreter(), interpreter()
        coral.invoke.side_effect = [None, RuntimeError("USB desconectado")]
        self.make_interpreter.side_effect = [coral, cpu]
        classifier = self.classifier(backend="coral", fallback_to_cpu=True)
        scores = classifier.predict_scores(np.zeros(YAMNET_EXPECTED_SAMPLES), source_sr=16000)
        self.assertEqual(scores.shape, (521,))
        cpu.invoke.assert_called_once()
        self.assertEqual(classifier.status()["active_backend"], "cpu")
        self.assertEqual(classifier.status()["fallbacks"], 1)
        self.assertEqual(classifier.status()["inference_count"], 1)

    def test_unknown_backend_is_rejected_by_config_validation(self):
        config = load_config(ROOT / "config.example.yaml")
        config["audio_classifier"]["backend"] = "auto"
        with self.assertRaisesRegex(ConfigError, "audio_classifier.backend"):
            validate_config(config)
