#!/usr/bin/env python3
"""
Test del micrófono Dayton iMM-6C (USB-C).
Verifica detección, captura y nivel de señal.
"""

import sys
import numpy as np
import sounddevice as sd


def find_imm6c():
    """Busca el dispositivo iMM-6C en la lista de audio."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if "iMM-6C" in d["name"] or "iMM6C" in d["name"]:
            return i, d
    return None, None


def test_audio(duration=2.0, sample_rate=44100):
    print("=== Dispositivos de audio ===")
    print(sd.query_devices())
    print()

    idx, dev = find_imm6c()
    if idx is None:
        print("ERROR: Dayton iMM-6C no encontrado")
        return False

    print(f"iMM-6C encontrado: [{idx}] {dev['name']}")
    print(f"  Canales entrada: {dev['max_input_channels']}")
    print(f"  Sample rate default: {dev['default_samplerate']}")
    print()

    print(f"Grabando {duration}s a {sample_rate} Hz...")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        device=idx,
        dtype="float32",
    )
    sd.wait()

    rms = np.sqrt(np.mean(recording**2))
    peak = np.max(np.abs(recording))
    db_rms = 20 * np.log10(rms / 20e-6) if rms > 0 else -np.inf
    db_peak = 20 * np.log10(peak / 20e-6) if peak > 0 else -np.inf

    print(f"Muestras: {len(recording)}")
    print(f"RMS: {rms:.6f} ({db_rms:.1f} dB sin calibrar)")
    print(f"Peak: {peak:.6f} ({db_peak:.1f} dB sin calibrar)")

    if rms > 0 and len(recording) == int(duration * sample_rate):
        print("RESULTADO: OK")
        return True
    else:
        print("RESULTADO: FALLO")
        return False


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    success = test_audio(duration=duration)
    sys.exit(0 if success else 1)
