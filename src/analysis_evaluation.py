"""Comparación offline con anotaciones, agrupada por condición de escucha."""

from collections import Counter, defaultdict

from scipy.stats import spearmanr


def classification_report(pairs: list[tuple[str, str]]) -> dict:
    matrix = defaultdict(Counter)
    for expected, predicted in pairs:
        matrix[expected][predicted] += 1
    metrics = {}
    for label in sorted({label for pair in pairs for label in pair}):
        correct = matrix[label][label]
        predicted_count = sum(row[label] for row in matrix.values())
        expected_count = matrix[label].total()
        metrics[label] = {
            "precision": correct / predicted_count if predicted_count else None,
            "recall": correct / expected_count if expected_count else None,
            "false_positives": predicted_count - correct,
            "support": expected_count,
        }
    return {"events": len(pairs), "confusion_matrix": {key: dict(row) for key, row in matrix.items() if row},
            "metrics": metrics}


def rank_correlation(values, ratings) -> float | None:
    if len(values) < 3 or len(set(values)) <= 1 or len(set(ratings)) <= 1:
        return None
    return float(spearmanr(values, ratings).statistic)


def summarize(results: list[dict]) -> dict:
    spatial, labels, coverage = [], [], Counter()
    groups = defaultdict(list)
    for row in results:
        review = row["human_review"]
        if row["status"] == "completed":
            predicted = row["classification"]
            if review.get("expected_attribution"):
                spatial.append((review["expected_attribution"], predicted["source_attribution"]))
            if review.get("expected_yamnet_label"):
                audio = predicted["audio_labels"]
                expected = review["expected_yamnet_label"]
                labels.append((expected, audio[0]["label"] if audio else "sin etiqueta"))
                coverage["evaluated"] += 1
                coverage["expected_emitted"] += any(item["label"] == expected for item in audio)
        if review.get("listening_condition") and review.get("annoyance_rating") is not None:
            condition = review["listening_condition"]
            groups[(condition, None)].append(row)
            if review.get("category"):
                groups[(condition, review["category"])].append(row)

    comparisons = []
    for (condition, category), group in groups.items():
        metrics = {}
        for key in ("loudness_n5_sone", "sharpness_acum", "roughness_asper", "tonality", "model_annoyance_score"):
            paired = [
                (row["audio"]["laeq_dba"], row["psychoacoustics"][key], row["human_review"]["annoyance_rating"])
                for row in group
                if row["audio"]["laeq_dba"] is not None and row["psychoacoustics"][key] is not None
            ]
            baseline, values, ratings = zip(*paired) if paired else ([], [], [])
            metrics[key] = {
                "events": len(paired), "spearman": rank_correlation(values, ratings),
                "laeq_spearman_same_events": rank_correlation(baseline, ratings),
            }
        comparisons.append({"listening_condition": condition, "category": category, "metrics": metrics})

    return {
        "attribution": classification_report(spatial),
        "yamnet_top_label": classification_report(labels),
        "yamnet_expected_label_coverage": dict(coverage),
        "annoyance_rank_comparisons": comparisons,
        "processing_status": dict(Counter(row["status"] for row in results)),
    }
