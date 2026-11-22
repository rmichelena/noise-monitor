"""Comparación offline con anotaciones, agrupada por condición de escucha."""

from collections import Counter, defaultdict
from itertools import combinations

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


def pairwise_rank_error(values, ratings) -> dict:
    compared = incorrect = tied = 0
    for (left_value, left_rating), (right_value, right_rating) in combinations(zip(values, ratings), 2):
        if left_rating == right_rating:
            continue
        compared += 1
        if left_value == right_value:
            tied += 1
        elif (left_value > right_value) != (left_rating > right_rating):
            incorrect += 1
    return {"pairs": compared, "inverted": incorrect, "tied_predictions": tied}


def summarize(results: list[dict]) -> dict:
    spatial, labels, raw_labels, coverage = [], [], [], Counter()
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
                top = audio[0]["label"] if audio else "sin etiqueta"
                labels.append((expected, top))
                raw_labels.append((expected, predicted.get("raw_yamnet_top_label", top) or "sin etiqueta"))
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
        for key in ("loudness_n5_sone", "sharpness_acum", "roughness_asper", "tonality",
                    "fluctuation_strength_vacil", "model_annoyance_score"):
            paired = [
                (row["audio"]["laeq_dba"], row["psychoacoustics"][key], row["human_review"]["annoyance_rating"])
                for row in group
                if row["audio"]["laeq_dba"] is not None and row["psychoacoustics"].get(key) is not None
            ]
            baseline, values, ratings = zip(*paired) if paired else ([], [], [])
            metrics[key] = {
                "events": len(paired), "spearman": rank_correlation(values, ratings),
                "laeq_spearman_same_events": rank_correlation(baseline, ratings),
                "rank_errors": pairwise_rank_error(values, ratings),
                "laeq_rank_errors_same_events": pairwise_rank_error(baseline, ratings),
            }
        comparisons.append({"listening_condition": condition, "category": category, "metrics": metrics})

    return {
        "attribution": classification_report(spatial),
        "yamnet_top_label": classification_report(labels),
        "yamnet_raw_top_label": classification_report(raw_labels),
        "yamnet_expected_label_coverage": dict(coverage),
        "annoyance_rank_comparisons": comparisons,
        "processing_status": dict(Counter(row["status"] for row in results)),
    }
