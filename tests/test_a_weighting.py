#!/usr/bin/env python3
"""
Regresión: la curva A-weighting debe cumplir IEC 61672-1 (0 dB a 1 kHz).

El filtro se diseñaba con el factor de normalización invertido
(10^(-2/20) en vez de 10^(+2/20)): la forma de la curva era correcta pero
todo el nivel caía 4.0 dB, así que cada dB(A) guardado como evidencia estaba
4 dB por debajo del valor real.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.signal import freqz

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.utils import (  # noqa: E402
    MIN_DB,
    apply_a_weighting,
    design_a_weighting,
    rms_to_db,
)


def response_db(fs: float, freqs: list[float]) -> np.ndarray:
    b, a = design_a_weighting(fs)
    _, h = freqz(b, a, worN=freqs, fs=fs)
    return 20.0 * np.log10(np.abs(h))


class AWeightingFilterTest(unittest.TestCase):
    def test_reference_frequency_is_zero_db(self):
        # IEC 61672-1 define A(1000 Hz) = 0 dB por construcción.
        self.assertAlmostEqual(float(response_db(44100, [1000.0])[0]), 0.0, delta=0.1)
        self.assertAlmostEqual(float(response_db(16000, [1000.0])[0]), 0.0, delta=0.1)

    def test_100_hz_matches_the_standard(self):
        # Tabla del estándar: -19.1 dB a 100 Hz (antes daba -23.1).
        self.assertAlmostEqual(float(response_db(44100, [100.0])[0]), -19.1, delta=0.4)

    def test_20_hz_and_2_khz(self):
        got = response_db(44100, [20.0, 2000.0])
        self.assertAlmostEqual(float(got[0]), -50.4, delta=0.6)
        self.assertAlmostEqual(float(got[1]), 1.2, delta=0.4)

    def test_high_frequency_rolloff(self):
        # El bilineal sin prewarp deforma el extremo alto: se acepta ±2 dB.
        got = float(response_db(44100, [10000.0])[0])
        self.assertLess(got, -1.5)
        self.assertGreater(got, -5.0)

    def test_coefficients_are_cached(self):
        self.assertIs(design_a_weighting(22050), design_a_weighting(22050))

    def test_invalid_sample_rate(self):
        with self.assertRaises(ValueError):
            design_a_weighting(0)

    def test_filtered_tone_attenuates_100_hz_by_19_db(self):
        fs = 44100
        t = np.arange(fs, dtype=np.float32) / fs
        tone = (0.5 * np.sin(2 * np.pi * 100.0 * t)).astype(np.float32)
        out = apply_a_weighting(tone, fs)
        steady = slice(fs // 2, None)  # descartar el transitorio del filtro
        rms_in = float(np.sqrt(np.mean(tone[steady] ** 2)))
        rms_out = float(np.sqrt(np.mean(out[steady] ** 2)))
        self.assertAlmostEqual(20 * np.log10(rms_out / rms_in), -19.1, delta=0.5)


class RmsToDbTest(unittest.TestCase):
    def test_full_scale_is_94_db_spl(self):
        # 1 Pa RMS = 94 dB SPL sobre 20 µPa.
        self.assertAlmostEqual(rms_to_db(1.0), 94.0, delta=0.1)

    def test_silence_returns_the_floor_not_minus_infinity(self):
        self.assertEqual(rms_to_db(0.0), MIN_DB)
        self.assertEqual(rms_to_db(1e-12), MIN_DB)
        self.assertTrue(np.isfinite(rms_to_db(0.0)))

    def test_nan_and_inf_are_floored(self):
        self.assertEqual(rms_to_db(float("nan")), MIN_DB)
        self.assertEqual(rms_to_db(float("inf")), MIN_DB)
        self.assertEqual(rms_to_db(float("-inf")), MIN_DB)

    def test_quiet_signal_still_measures(self):
        # -60 dBFS ≈ 34 dB SPL: por encima del piso, no se recorta.
        self.assertAlmostEqual(rms_to_db(0.001), 33.98, delta=0.1)


if __name__ == "__main__":
    unittest.main()
