"""Calibration and metrics for the released ReDeEP token score."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .data import TASKS


def binary_auroc(
    labels: Sequence[int],
    scores: Sequence[float],
) -> float:
    """Compute AUROC from average ranks, including tied scores."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores have different lengths")
    if not labels:
        raise ValueError("AUROC requires at least one observation")
    normalized_labels: list[int] = []
    normalized_scores: list[float] = []
    for index, (label, score) in enumerate(zip(labels, scores)):
        if label not in (0, 1):
            raise ValueError(f"label {index} is not binary")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError(f"score {index} is not finite")
        normalized_labels.append(int(label))
        normalized_scores.append(numeric_score)

    positive_count = sum(normalized_labels)
    negative_count = len(normalized_labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC requires both classes")

    ordered = sorted(
        enumerate(normalized_scores),
        key=lambda item: item[1],
    )
    ranks = [0.0] * len(ordered)
    start = 0
    while start < len(ordered):
        end = start + 1
        while (
            end < len(ordered)
            and ordered[end][1] == ordered[start][1]
        ):
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for ordered_index in range(start, end):
            original_index = ordered[ordered_index][0]
            ranks[original_index] = average_rank
        start = end

    positive_rank_sum = sum(
        rank
        for rank, label in zip(ranks, normalized_labels)
        if label == 1
    )
    mann_whitney = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    )
    return mann_whitney / (positive_count * negative_count)


def _validate_feature_rows(
    rows: Sequence[Mapping[str, Any]],
    candidate_count: int,
) -> int:
    if not rows:
        raise ValueError("calibration requires at least one response")
    layer_count: int | None = None
    for response_index, row in enumerate(rows):
        labels = row.get("labels")
        external = row.get("external")
        parametric = row.get("parametric")
        if not isinstance(labels, Sequence):
            raise ValueError(f"row {response_index} has invalid labels")
        if (
            not isinstance(external, Sequence)
            or not isinstance(parametric, Sequence)
            or len(external) != len(labels)
            or len(parametric) != len(labels)
        ):
            raise ValueError(
                f"row {response_index} has inconsistent token dimensions"
            )
        for token_index, (external_row, parametric_row) in enumerate(
            zip(external, parametric)
        ):
            if len(external_row) != candidate_count:
                raise ValueError(
                    f"row {response_index} token {token_index} has "
                    "the wrong ECS dimension"
                )
            current_layer_count = len(parametric_row)
            if layer_count is None:
                layer_count = current_layer_count
            elif current_layer_count != layer_count:
                raise ValueError("PKS dimensions are inconsistent")
    if layer_count is None or layer_count == 0:
        raise ValueError("no parametric layers were provided")
    return layer_count


def _flatten_tokens(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[Sequence[float]], list[Sequence[float]]]:
    labels: list[int] = []
    external: list[Sequence[float]] = []
    parametric: list[Sequence[float]] = []
    for row in rows:
        labels.extend(int(value) for value in row["labels"])
        external.extend(row["external"])
        parametric.extend(row["parametric"])
    return labels, external, parametric


def _rank_features(
    feature_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    positive_label: int,
) -> list[tuple[float, int]]:
    if not feature_rows:
        raise ValueError("cannot rank empty features")
    feature_count = len(feature_rows[0])
    target = (
        list(labels)
        if positive_label == 1
        else [1 - int(label) for label in labels]
    )
    try:
        import numpy
    except ImportError:
        numpy = None
    if numpy is not None:
        matrix = numpy.asarray(feature_rows, dtype=numpy.float64)
        target_array = numpy.asarray(target, dtype=numpy.int8)
        positive_count = int(target_array.sum())
        negative_count = int(target_array.size - positive_count)
        if positive_count == 0 or negative_count == 0:
            raise ValueError("AUROC requires both classes")
        ranked = []
        for feature_index in range(matrix.shape[1]):
            scores = matrix[:, feature_index]
            order = numpy.argsort(scores, kind="mergesort")
            ordered_scores = scores[order]
            ordered_labels = target_array[order]
            boundaries = numpy.flatnonzero(
                numpy.r_[
                    True,
                    ordered_scores[1:] != ordered_scores[:-1],
                    True,
                ]
            )
            starts = boundaries[:-1]
            ends = boundaries[1:]
            average_ranks = (starts + 1 + ends) / 2.0
            ordered_ranks = numpy.repeat(
                average_ranks, ends - starts
            )
            positive_rank_sum = float(
                ordered_ranks[ordered_labels == 1].sum()
            )
            mann_whitney = (
                positive_rank_sum
                - positive_count * (positive_count + 1) / 2.0
            )
            ranked.append(
                (
                    mann_whitney
                    / (positive_count * negative_count),
                    feature_index,
                )
            )
        return sorted(ranked, key=lambda item: (-item[0], item[1]))

    ranked = []
    for feature_index in range(feature_count):
        scores = [
            float(feature_row[feature_index])
            for feature_row in feature_rows
        ]
        ranked.append(
            (binary_auroc(target, scores), feature_index)
        )
    return sorted(ranked, key=lambda item: (-item[0], item[1]))


def _minmax(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot fit a range on no values")
    minimum = min(values)
    maximum = max(values)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("calibration features contain non-finite values")
    if maximum == minimum:
        raise ValueError("calibration feature range is zero")
    return minimum, maximum


def _normalize(value: float, bounds: Sequence[float]) -> float:
    minimum, maximum = float(bounds[0]), float(bounds[1])
    if maximum <= minimum:
        raise ValueError("invalid normalization range")
    return (float(value) - minimum) / (maximum - minimum)


def _score_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_head_indices: Sequence[int],
    selected_layers: Sequence[int],
    beta: float,
    external_range: Sequence[float],
    parametric_range: Sequence[float],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        token_scores: list[float] = []
        for external_row, parametric_row in zip(
            row["external"], row["parametric"]
        ):
            external_sum = sum(
                float(external_row[index])
                for index in selected_head_indices
            )
            parametric_sum = sum(
                float(parametric_row[index])
                for index in selected_layers
            )
            token_scores.append(
                _normalize(parametric_sum, parametric_range)
                - float(beta)
                * _normalize(external_sum, external_range)
            )
        scored.append({**row, "scores": token_scores})
    return scored


def calibrate_redeep(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_heads: Sequence[Sequence[int]],
    selection_unit: str,
    head_counts: Iterable[int],
    layer_counts: Iterable[int],
    beta_values: Iterable[float],
) -> dict[str, Any]:
    """Rank components and select the released linear score on held-out data."""

    if selection_unit not in {"token", "response"}:
        raise ValueError("selection_unit must be 'token' or 'response'")
    normalized_heads = [
        (int(head[0]), int(head[1])) for head in candidate_heads
    ]
    layer_count = _validate_feature_rows(rows, len(normalized_heads))
    token_labels, external_rows, parametric_rows = _flatten_tokens(rows)

    if selection_unit == "token":
        ranking_labels = token_labels
        ranking_external = external_rows
        ranking_parametric = parametric_rows
    else:
        ranking_labels = [
            int(row.get("response_label", max(row["labels"], default=0)))
            for row in rows
        ]
        ranking_external = [
            [
                sum(float(token[index]) for token in row["external"])
                / len(row["external"])
                for index in range(len(normalized_heads))
            ]
            for row in rows
        ]
        ranking_parametric = [
            [
                sum(float(token[index]) for token in row["parametric"])
                / len(row["parametric"])
                for index in range(layer_count)
            ]
            for row in rows
        ]

    head_ranking = _rank_features(
        ranking_external,
        ranking_labels,
        positive_label=0,
    )
    layer_ranking = _rank_features(
        ranking_parametric,
        ranking_labels,
        positive_label=1,
    )

    normalized_head_counts = sorted(set(int(value) for value in head_counts))
    normalized_layer_counts = sorted(
        set(int(value) for value in layer_counts)
    )
    normalized_betas = sorted(set(float(value) for value in beta_values))
    if (
        not normalized_head_counts
        or not normalized_layer_counts
        or not normalized_betas
    ):
        raise ValueError("all calibration grids must be non-empty")
    if (
        normalized_head_counts[0] <= 0
        or normalized_head_counts[-1] > len(normalized_heads)
    ):
        raise ValueError("head count lies outside the candidate set")
    if (
        normalized_layer_counts[0] <= 0
        or normalized_layer_counts[-1] > layer_count
    ):
        raise ValueError("layer count lies outside the model")
    if normalized_betas[0] <= 0:
        raise ValueError("beta must be positive")

    best: dict[str, Any] | None = None
    for head_count in normalized_head_counts:
        selected_head_indices = [
            index for _auc, index in head_ranking[:head_count]
        ]
        external_sums = [
            sum(float(values[index]) for index in selected_head_indices)
            for values in external_rows
        ]
        external_range = _minmax(external_sums)
        for selected_layer_count in normalized_layer_counts:
            selected_layers = [
                index
                for _auc, index in layer_ranking[:selected_layer_count]
            ]
            parametric_sums = [
                sum(float(values[index]) for index in selected_layers)
                for values in parametric_rows
            ]
            parametric_range = _minmax(parametric_sums)
            for beta in normalized_betas:
                token_scores = [
                    _normalize(parametric_sum, parametric_range)
                    - beta * _normalize(external_sum, external_range)
                    for external_sum, parametric_sum in zip(
                        external_sums, parametric_sums
                    )
                ]
                if selection_unit == "token":
                    objective_labels = token_labels
                    objective_scores = token_scores
                else:
                    objective_labels = ranking_labels
                    objective_scores = []
                    cursor = 0
                    for row in rows:
                        token_count = len(row["labels"])
                        response_scores = token_scores[
                            cursor : cursor + token_count
                        ]
                        cursor += token_count
                        objective_scores.append(
                            sum(response_scores) / len(response_scores)
                        )
                objective_auc = binary_auroc(
                    objective_labels, objective_scores
                )
                candidate = {
                    "selection_unit": selection_unit,
                    "selected_head_indices": selected_head_indices,
                    "selected_heads": [
                        list(normalized_heads[index])
                        for index in selected_head_indices
                    ],
                    "selected_layers": selected_layers,
                    "beta": beta,
                    "external_range": list(external_range),
                    "parametric_range": list(parametric_range),
                    "calibration_auroc": objective_auc,
                }
                candidate_key = (
                    objective_auc,
                    -head_count,
                    -selected_layer_count,
                    -beta,
                )
                if (
                    best is None
                    or candidate_key > best["_selection_key"]
                ):
                    best = {
                        **candidate,
                        "_selection_key": candidate_key,
                    }
    assert best is not None
    best.pop("_selection_key")
    best["candidate_heads"] = [
        list(head) for head in normalized_heads
    ]
    best["head_ranking"] = [
        {
            "candidate_index": index,
            "head": list(normalized_heads[index]),
            "auroc_against_truth": auc,
        }
        for auc, index in head_ranking
    ]
    best["layer_ranking"] = [
        {
            "layer": index,
            "auroc_against_hallucination": auc,
        }
        for auc, index in layer_ranking
    ]
    return best


def apply_calibration(
    rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return _score_rows(
        rows,
        selected_head_indices=calibration["selected_head_indices"],
        selected_layers=calibration["selected_layers"],
        beta=float(calibration["beta"]),
        external_range=calibration["external_range"],
        parametric_range=calibration["parametric_range"],
    )


def _metric_block(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    return {
        "auroc": binary_auroc(labels, scores),
        "num_tokens": len(labels),
        "num_hallucinated_tokens": sum(int(label) for label in labels),
        "num_truthful_tokens": len(labels) - sum(int(label) for label in labels),
    }


def compute_token_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str] = TASKS,
) -> dict[str, Any]:
    pooled_labels: list[int] = []
    pooled_scores: list[float] = []
    task_values: dict[str, tuple[list[int], list[float]]] = {
        task: ([], []) for task in tasks
    }
    for row in rows:
        task = str(row["task_type"])
        if task not in task_values:
            continue
        labels = [int(value) for value in row["labels"]]
        scores = [float(value) for value in row["scores"]]
        if len(labels) != len(scores):
            raise ValueError(f"row {row.get('id')} has score/label mismatch")
        pooled_labels.extend(labels)
        pooled_scores.extend(scores)
        task_values[task][0].extend(labels)
        task_values[task][1].extend(scores)

    per_task = {
        task: _metric_block(*task_values[task])
        for task in tasks
    }
    task_aurocs = [per_task[task]["auroc"] for task in tasks]
    task_supports = [per_task[task]["num_tokens"] for task in tasks]
    total_support = sum(task_supports)
    return {
        "overall": _metric_block(pooled_labels, pooled_scores),
        "per_task": per_task,
        "task_macro_auroc": sum(task_aurocs) / len(task_aurocs),
        "support_weighted_task_auroc": sum(
            auc * support
            for auc, support in zip(task_aurocs, task_supports)
        )
        / total_support,
    }


def _response_metric_block(
    labels: Sequence[int],
    scores: Sequence[float],
) -> dict[str, Any]:
    hallucinated = sum(int(label) for label in labels)
    return {
        "auroc": binary_auroc(labels, scores),
        "num_responses": len(labels),
        "num_hallucinated_responses": hallucinated,
        "num_truthful_responses": len(labels) - hallucinated,
    }


def compute_response_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str] = TASKS,
) -> dict[str, Any]:
    """Sanity-check the paper's mean-token response readout."""

    pooled_labels: list[int] = []
    pooled_scores: list[float] = []
    task_values: dict[str, tuple[list[int], list[float]]] = {
        task: ([], []) for task in tasks
    }
    for row in rows:
        task = str(row["task_type"])
        if task not in task_values:
            continue
        labels = [int(value) for value in row["labels"]]
        scores = [float(value) for value in row["scores"]]
        if not labels or len(labels) != len(scores):
            raise ValueError(
                f"row {row.get('id')} has invalid response scores"
            )
        response_label = int(max(labels))
        response_score = sum(scores) / len(scores)
        pooled_labels.append(response_label)
        pooled_scores.append(response_score)
        task_values[task][0].append(response_label)
        task_values[task][1].append(response_score)

    per_task = {
        task: _response_metric_block(*task_values[task])
        for task in tasks
    }
    task_aurocs = [per_task[task]["auroc"] for task in tasks]
    return {
        "overall": _response_metric_block(pooled_labels, pooled_scores),
        "per_task": per_task,
        "task_macro_auroc": sum(task_aurocs) / len(task_aurocs),
    }
