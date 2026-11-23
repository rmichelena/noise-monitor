#!/usr/bin/env python3
"""Compara los backends de YAMNet con los mismos clips y ventanas del monitor."""

import argparse
import json
import math
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.acoustic_analysis import file_sha256, read_clip  # noqa: E402
from src.audio_classifier import AudioClassifier  # noqa: E402
from src.utils import load_config  # noqa: E402


def paths_for(config: dict) -> dict:
    cpu = ROOT / config["model_path"]
    return {
        "cpu": cpu,
        "coral": ROOT / config.get("coral_model_path", cpu.with_name("yamnet_edgetpu.tflite")),
        "labels": ROOT / config["labels_path"],
    }


def compare(manifest: Path, config: dict, backends: list[str], repeats: int, scan_all_speech: bool) -> dict:
    paths = paths_for(config)
    classifiers = {
        backend: AudioClassifier(
            paths["cpu"], paths["labels"], config.get("category_map", {}),
            min_score=float(config.get("min_score", .3)), backend=backend,
            coral_model_path=paths["coral"], fallback_to_cpu=False,
        )
        for backend in backends
    }
    clips = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(clips, list) or not clips:
        raise ValueError("el manifiesto debe contener al menos un clip")
    if len({clip["id"] for clip in clips}) != len(clips):
        raise ValueError("los ids de clips deben ser únicos")
    timings = {backend: {"wall": [], "cpu": []} for backend in backends}
    results = []
    for clip in clips:
        samples, rate, _, _ = read_clip((manifest.parent / clip["clip"]).resolve())
        trigger = float(clip["trigger_seconds"])
        if not math.isfinite(trigger) or not 0 <= trigger <= len(samples) / rate:
            raise ValueError(f"trigger_seconds fuera del clip {clip['id']}")
        if "speech" in clip and not isinstance(clip["speech"], bool):
            raise ValueError(f"speech debe ser true o false en {clip['id']}")
        expected = clip.get("expected_labels", [])
        if not isinstance(expected, list) or any(not isinstance(label, str) for label in expected):
            raise ValueError(f"expected_labels debe ser una lista de etiquetas en {clip['id']}")
        row = {"id": clip["id"], "speech": clip.get("speech"), "expected_labels": expected, "results": {}}
        for backend, classifier in classifiers.items():
            anchor = round(trigger * rate)
            if not results:  # Warm up resampy and the interpreter before measuring.
                classifier.classify_event(samples, rate, anchor, scan_all_speech=scan_all_speech)
            for _ in range(repeats):
                started = perf_counter()
                cpu_started = process_time()
                labels, speech = classifier.classify_event(samples, rate, anchor, scan_all_speech=scan_all_speech)
                timings[backend]["cpu"].append((process_time() - cpu_started) * 1000)
                timings[backend]["wall"].append((perf_counter() - started) * 1000)
            row["results"][backend] = {
                "primary": labels[0]["label"] if labels and labels[0]["passes_threshold"] else "unknown",
                "labels": labels, "muted": speech,
            }
        results.append(row)

    summary = {}
    for backend in backends:
        summary[backend] = {
            "model_sha256": file_sha256(paths[backend]),
            "classification_wall_ms_p50": round(float(np.median(timings[backend]["wall"])), 2),
            "classification_wall_ms_p95": round(float(np.percentile(timings[backend]["wall"], 95)), 2),
            "classification_cpu_ms_p50": round(float(np.median(timings[backend]["cpu"])), 2),
            "classification_cpu_ms_p95": round(float(np.percentile(timings[backend]["cpu"], 95)), 2),
            "unknown": sum(row["results"][backend]["primary"] == "unknown" for row in results),
            "annotated_labels": sum(bool(row["expected_labels"]) for row in results),
            "top3_hits_at_threshold": sum(
                any(tag["passes_threshold"] and tag["label"] in row["expected_labels"]
                    for tag in row["results"][backend]["labels"]) for row in results
            ),
            "speech_positives": sum(row["speech"] is True for row in results),
            "speech_missed": [row["id"] for row in results if row["speech"] is True and not row["results"][backend]["muted"]],
            "false_mutes": [row["id"] for row in results if row["speech"] is False and row["results"][backend]["muted"]],
            "runtime": classifiers[backend].status(),
        }
    if len(backends) == 2:
        summary["parity"] = {
            "primary_agreement": sum(row["results"]["cpu"]["primary"] == row["results"]["coral"]["primary"]
                                     for row in results),
            "muted_by_cpu_only": [row["id"] for row in results
                                  if row["results"]["cpu"]["muted"] and not row["results"]["coral"]["muted"]],
        }
    return {
        "mode": "psychoacoustic" if scan_all_speech else "dba", "clips": len(results), "repeats": repeats,
        "min_score": float(config.get("min_score", .3)), "python": sys.version.split()[0],
        "litert": version("ai-edge-litert"), "labels_sha256": file_sha256(paths["labels"]),
        "summary": summary, "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--backend", choices=("both", "cpu", "coral"), default="both")
    parser.add_argument("--mode", choices=("dba", "psychoacoustic"))
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat debe ser mayor que cero")
    try:
        config = load_config(args.config)
        mode = args.mode or config["triggers"].get("criterion", "dba")
        if mode not in ("dba", "psychoacoustic"):
            raise ValueError("triggers.criterion debe ser dba o psychoacoustic")
        report = compare(args.manifest, config["audio_classifier"],
                         ["cpu", "coral"] if args.backend == "both" else [args.backend],
                         args.repeat, mode == "psychoacoustic")
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Benchmark YAMNet falló ({args.backend}): {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
