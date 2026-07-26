"""Token- and response-level evaluation for ReDeEP feature tables."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_TASKS = ("QA", "Data2txt", "Summary")


@dataclass(frozen=True)
class _ResponseCluster:
    key: tuple[str, str]
    token_labels: np.ndarray
    token_scores: np.ndarray
    response_label: int
    response_score: float


@dataclass(frozen=True)
class _PreparedClusters:
    clusters: tuple[_ResponseCluster, ...]
    token_labels: np.ndarray
    token_scores: np.ndarray
    token_cluster_index: np.ndarray
    response_labels: np.ndarray
    response_scores: np.ndarray


def safe_roc_auc(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return ROC-AUC or NaN for empty/single-class data instead of raising."""

    label_array = np.asarray(labels)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.shape != score_array.shape:
        raise ValueError("labels and scores must have the same shape")
    if sample_weight is None:
        weight_array = np.ones(score_array.shape, dtype=np.float64)
    else:
        weight_array = np.asarray(sample_weight, dtype=np.float64)
        if weight_array.shape != score_array.shape:
            raise ValueError("sample_weight must have the same shape as labels")
    valid = (
        np.isfinite(score_array)
        & np.isfinite(weight_array)
        & (weight_array > 0)
        & pd.notna(label_array)
    )
    if not valid.any():
        return float("nan")
    selected_labels = label_array[valid].astype(np.int8, copy=False)
    selected_scores = score_array[valid]
    selected_weights = weight_array[valid]
    if np.unique(selected_labels).size < 2:
        return float("nan")
    return float(
        roc_auc_score(
            selected_labels,
            selected_scores,
            sample_weight=selected_weights,
        )
    )


def evaluate_model(
    scored_tokens: pd.DataFrame,
    *,
    model_name: str,
    mode: str,
    score_column: str = "redeep_score",
    tasks: Sequence[str] = DEFAULT_TASKS,
    n_bootstrap: int = 1_000,
    confidence: float = 0.95,
    seed: int = 42,
    task_column: str = "task",
    response_column: str = "response_id",
    token_label_column: str = "token_label",
    response_label_column: str = "response_label",
) -> pd.DataFrame:
    """Report token micro-AUC and paper-style response mean-token AUC.

    Confidence intervals resample response clusters. A response drawn multiple
    times receives the corresponding sample weight, so duplicate draws are not
    accidentally collapsed by a later group-by.
    """

    _validate_bootstrap(n_bootstrap, confidence)
    _require_columns(
        scored_tokens,
        [
            task_column,
            response_column,
            token_label_column,
            response_label_column,
            score_column,
        ],
    )
    _validate_binary_column(scored_tokens[token_label_column], token_label_column)
    _validate_binary_column(scored_tokens[response_label_column], response_label_column)
    _validate_finite_scores(scored_tokens[score_column], score_column)

    records: list[dict[str, object]] = []
    scopes: list[tuple[str, set[str] | None]] = [
        *((str(task), {str(task)}) for task in tasks),
        ("Overall", None),
    ]
    for scope_name, task_filter in scopes:
        prepared = _prepare_clusters(
            scored_tokens,
            task_filter=task_filter,
            task_column=task_column,
            response_column=response_column,
            token_label_column=token_label_column,
            response_label_column=response_label_column,
            score_column=score_column,
        )
        token_auc = safe_roc_auc(prepared.token_labels, prepared.token_scores)
        response_auc = safe_roc_auc(prepared.response_labels, prepared.response_scores)
        bootstrap = _bootstrap_metrics(
            prepared,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=_scope_seed(seed, scope_name),
        )
        positive_tokens = int(prepared.token_labels.sum())
        positive_responses = int(prepared.response_labels.sum())
        records.append(
            {
                "model": model_name,
                "mode": mode,
                "task": scope_name,
                "n_responses": len(prepared.clusters),
                "n_tokens": int(prepared.token_labels.size),
                "n_positive_tokens": positive_tokens,
                "n_negative_tokens": int(prepared.token_labels.size - positive_tokens),
                "n_positive_responses": positive_responses,
                "n_negative_responses": int(
                    prepared.response_labels.size - positive_responses
                ),
                "token_micro_roc_auc": token_auc,
                "token_auc_ci_low": bootstrap["token_ci_low"],
                "token_auc_ci_high": bootstrap["token_ci_high"],
                "token_bootstrap_valid": bootstrap["token_valid"],
                "paper_response_mean_token_roc_auc": response_auc,
                "response_auc_ci_low": bootstrap["response_ci_low"],
                "response_auc_ci_high": bootstrap["response_ci_high"],
                "response_bootstrap_valid": bootstrap["response_valid"],
                "bootstrap_samples": n_bootstrap,
                "confidence": confidence,
            }
        )

    # TaskMacro is intentionally bootstrapped as a statistic: every replicate
    # resamples response clusters independently within each task, computes the
    # three task AUCs, and only then averages them. Averaging the endpoints of
    # three independently computed confidence intervals would be incorrect.
    task_prepared = [
        _prepare_clusters(
            scored_tokens,
            task_filter={str(task)},
            task_column=task_column,
            response_column=response_column,
            token_label_column=token_label_column,
            response_label_column=response_label_column,
            score_column=score_column,
        )
        for task in tasks
    ]
    combined = _prepare_cluster_sequence(
        tuple(
            cluster
            for prepared in task_prepared
            for cluster in prepared.clusters
        )
    )
    task_token_auc = [
        safe_roc_auc(prepared.token_labels, prepared.token_scores)
        for prepared in task_prepared
    ]
    task_response_auc = [
        safe_roc_auc(prepared.response_labels, prepared.response_scores)
        for prepared in task_prepared
    ]
    token_macro = _strict_mean(task_token_auc)
    response_macro = _strict_mean(task_response_auc)
    macro_bootstrap = _bootstrap_macro_metrics(
        task_prepared,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=_scope_seed(seed, "TaskMacro"),
    )
    positive_tokens = int(combined.token_labels.sum())
    positive_responses = int(combined.response_labels.sum())
    records.append(
        {
            "model": model_name,
            "mode": mode,
            "task": "TaskMacro",
            "n_responses": len(combined.clusters),
            "n_tokens": int(combined.token_labels.size),
            "n_positive_tokens": positive_tokens,
            "n_negative_tokens": int(combined.token_labels.size - positive_tokens),
            "n_positive_responses": positive_responses,
            "n_negative_responses": int(
                combined.response_labels.size - positive_responses
            ),
            "token_micro_roc_auc": token_macro,
            "token_auc_ci_low": macro_bootstrap["token_ci_low"],
            "token_auc_ci_high": macro_bootstrap["token_ci_high"],
            "token_bootstrap_valid": macro_bootstrap["token_valid"],
            "paper_response_mean_token_roc_auc": response_macro,
            "response_auc_ci_low": macro_bootstrap["response_ci_low"],
            "response_auc_ci_high": macro_bootstrap["response_ci_high"],
            "response_bootstrap_valid": macro_bootstrap["response_valid"],
            "bootstrap_samples": n_bootstrap,
            "confidence": confidence,
        }
    )
    return pd.DataFrame.from_records(records)


def compare_models_paired(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    first_model: str,
    second_model: str,
    first_score_column: str = "redeep_score",
    second_score_column: str = "redeep_score",
    tasks: Sequence[str] = DEFAULT_TASKS,
    n_bootstrap: int = 1_000,
    confidence: float = 0.95,
    seed: int = 42,
    task_column: str = "task",
    response_column: str = "response_id",
    token_label_column: str = "token_label",
    response_label_column: str = "response_label",
) -> pd.DataFrame:
    """Paired response-cluster bootstrap of ``first - second`` AUC.

    The two tokenizers may yield different token counts. Pairing is therefore
    performed at the shared response level, while each model retains its own
    token rows and labels within a sampled response.
    """

    _validate_bootstrap(n_bootstrap, confidence)
    for frame, score_column, name in (
        (first, first_score_column, first_model),
        (second, second_score_column, second_model),
    ):
        _require_columns(
            frame,
            [
                task_column,
                response_column,
                token_label_column,
                response_label_column,
                score_column,
            ],
        )
        _validate_binary_column(frame[token_label_column], f"{name}.{token_label_column}")
        _validate_binary_column(
            frame[response_label_column], f"{name}.{response_label_column}"
        )
        _validate_finite_scores(frame[score_column], f"{name}.{score_column}")

    first_clusters = _cluster_map(
        first,
        task_column=task_column,
        response_column=response_column,
        token_label_column=token_label_column,
        response_label_column=response_label_column,
        score_column=first_score_column,
    )
    second_clusters = _cluster_map(
        second,
        task_column=task_column,
        response_column=response_column,
        token_label_column=token_label_column,
        response_label_column=response_label_column,
        score_column=second_score_column,
    )

    records: list[dict[str, object]] = []
    scopes: list[tuple[str, set[str] | None]] = [
        *((str(task), {str(task)}) for task in tasks),
        ("Overall", None),
    ]
    for scope_name, task_filter in scopes:
        shared_keys = sorted(set(first_clusters).intersection(second_clusters))
        if task_filter is not None:
            shared_keys = [key for key in shared_keys if key[0] in task_filter]
        for key in shared_keys:
            if first_clusters[key].response_label != second_clusters[key].response_label:
                raise ValueError(f"Response-label mismatch between models for {key}")

        first_prepared = _prepare_cluster_sequence(
            tuple(first_clusters[key] for key in shared_keys)
        )
        second_prepared = _prepare_cluster_sequence(
            tuple(second_clusters[key] for key in shared_keys)
        )
        point_first_token = safe_roc_auc(
            first_prepared.token_labels, first_prepared.token_scores
        )
        point_second_token = safe_roc_auc(
            second_prepared.token_labels, second_prepared.token_scores
        )
        point_first_response = safe_roc_auc(
            first_prepared.response_labels, first_prepared.response_scores
        )
        point_second_response = safe_roc_auc(
            second_prepared.response_labels, second_prepared.response_scores
        )
        bootstrap = _paired_bootstrap(
            first_prepared,
            second_prepared,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=_scope_seed(seed, scope_name),
        )
        records.append(
            {
                "first_model": first_model,
                "second_model": second_model,
                "task": scope_name,
                "n_common_responses": len(shared_keys),
                "first_token_micro_roc_auc": point_first_token,
                "second_token_micro_roc_auc": point_second_token,
                "token_auc_difference": point_first_token - point_second_token,
                "token_difference_ci_low": bootstrap["token_ci_low"],
                "token_difference_ci_high": bootstrap["token_ci_high"],
                "token_bootstrap_valid": bootstrap["token_valid"],
                "first_paper_response_roc_auc": point_first_response,
                "second_paper_response_roc_auc": point_second_response,
                "response_auc_difference": point_first_response
                - point_second_response,
                "response_difference_ci_low": bootstrap["response_ci_low"],
                "response_difference_ci_high": bootstrap["response_ci_high"],
                "response_bootstrap_valid": bootstrap["response_valid"],
                "bootstrap_samples": n_bootstrap,
                "confidence": confidence,
            }
        )

    paired_by_task: list[tuple[_PreparedClusters, _PreparedClusters]] = []
    for task in tasks:
        task_keys = sorted(
            key
            for key in set(first_clusters).intersection(second_clusters)
            if key[0] == str(task)
        )
        for key in task_keys:
            if first_clusters[key].response_label != second_clusters[key].response_label:
                raise ValueError(f"Response-label mismatch between models for {key}")
        paired_by_task.append(
            (
                _prepare_cluster_sequence(
                    tuple(first_clusters[key] for key in task_keys)
                ),
                _prepare_cluster_sequence(
                    tuple(second_clusters[key] for key in task_keys)
                ),
            )
        )
    first_token_auc = [
        safe_roc_auc(prepared.token_labels, prepared.token_scores)
        for prepared, _ in paired_by_task
    ]
    second_token_auc = [
        safe_roc_auc(prepared.token_labels, prepared.token_scores)
        for _, prepared in paired_by_task
    ]
    first_response_auc = [
        safe_roc_auc(prepared.response_labels, prepared.response_scores)
        for prepared, _ in paired_by_task
    ]
    second_response_auc = [
        safe_roc_auc(prepared.response_labels, prepared.response_scores)
        for _, prepared in paired_by_task
    ]
    first_token_macro = _strict_mean(first_token_auc)
    second_token_macro = _strict_mean(second_token_auc)
    first_response_macro = _strict_mean(first_response_auc)
    second_response_macro = _strict_mean(second_response_auc)
    macro_bootstrap = _paired_macro_bootstrap(
        paired_by_task,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=_scope_seed(seed, "TaskMacro"),
    )
    records.append(
        {
            "first_model": first_model,
            "second_model": second_model,
            "task": "TaskMacro",
            "n_common_responses": sum(
                len(first_prepared.clusters)
                for first_prepared, _ in paired_by_task
            ),
            "first_token_micro_roc_auc": first_token_macro,
            "second_token_micro_roc_auc": second_token_macro,
            "token_auc_difference": first_token_macro - second_token_macro,
            "token_difference_ci_low": macro_bootstrap["token_ci_low"],
            "token_difference_ci_high": macro_bootstrap["token_ci_high"],
            "token_bootstrap_valid": macro_bootstrap["token_valid"],
            "first_paper_response_roc_auc": first_response_macro,
            "second_paper_response_roc_auc": second_response_macro,
            "response_auc_difference": first_response_macro - second_response_macro,
            "response_difference_ci_low": macro_bootstrap["response_ci_low"],
            "response_difference_ci_high": macro_bootstrap["response_ci_high"],
            "response_bootstrap_valid": macro_bootstrap["response_valid"],
            "bootstrap_samples": n_bootstrap,
            "confidence": confidence,
        }
    )
    return pd.DataFrame.from_records(records)


def _prepare_clusters(
    frame: pd.DataFrame,
    *,
    task_filter: set[str] | None,
    task_column: str,
    response_column: str,
    token_label_column: str,
    response_label_column: str,
    score_column: str,
) -> _PreparedClusters:
    if task_filter is not None:
        frame = frame.loc[frame[task_column].astype(str).isin(task_filter)]
    clusters = tuple(
        _cluster_map(
            frame,
            task_column=task_column,
            response_column=response_column,
            token_label_column=token_label_column,
            response_label_column=response_label_column,
            score_column=score_column,
        ).values()
    )
    return _prepare_cluster_sequence(clusters)


def _cluster_map(
    frame: pd.DataFrame,
    *,
    task_column: str,
    response_column: str,
    token_label_column: str,
    response_label_column: str,
    score_column: str,
) -> dict[tuple[str, str], _ResponseCluster]:
    clusters: dict[tuple[str, str], _ResponseCluster] = {}
    if frame.empty:
        return clusters
    group_columns = [task_column, response_column]
    for raw_key, group in frame.groupby(group_columns, sort=False, dropna=False):
        key = (str(raw_key[0]), str(raw_key[1]))
        response_labels = group[response_label_column].drop_duplicates().to_numpy()
        if response_labels.size != 1:
            raise ValueError(f"Inconsistent response labels within response {key}")
        token_labels = group[token_label_column].to_numpy(dtype=np.int8)
        token_scores = group[score_column].to_numpy(dtype=np.float64)
        clusters[key] = _ResponseCluster(
            key=key,
            token_labels=token_labels,
            token_scores=token_scores,
            response_label=int(response_labels[0]),
            response_score=float(token_scores.mean()),
        )
    return clusters


def _prepare_cluster_sequence(
    clusters: tuple[_ResponseCluster, ...],
) -> _PreparedClusters:
    if not clusters:
        return _PreparedClusters(
            clusters=(),
            token_labels=np.empty(0, dtype=np.int8),
            token_scores=np.empty(0, dtype=np.float64),
            token_cluster_index=np.empty(0, dtype=np.int64),
            response_labels=np.empty(0, dtype=np.int8),
            response_scores=np.empty(0, dtype=np.float64),
        )
    return _PreparedClusters(
        clusters=clusters,
        token_labels=np.concatenate([cluster.token_labels for cluster in clusters]),
        token_scores=np.concatenate([cluster.token_scores for cluster in clusters]),
        token_cluster_index=np.concatenate(
            [
                np.full(cluster.token_labels.size, index, dtype=np.int64)
                for index, cluster in enumerate(clusters)
            ]
        ),
        response_labels=np.asarray(
            [cluster.response_label for cluster in clusters], dtype=np.int8
        ),
        response_scores=np.asarray(
            [cluster.response_score for cluster in clusters], dtype=np.float64
        ),
    )


def _bootstrap_metrics(
    prepared: _PreparedClusters,
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    cluster_count = len(prepared.clusters)
    if cluster_count == 0 or n_bootstrap == 0:
        return _empty_bootstrap()
    rng = np.random.default_rng(seed)
    token_values: list[float] = []
    response_values: list[float] = []
    for _ in range(n_bootstrap):
        drawn = rng.integers(0, cluster_count, size=cluster_count)
        cluster_weights = np.bincount(drawn, minlength=cluster_count).astype(np.float64)
        token_weights = cluster_weights[prepared.token_cluster_index]
        token_auc = safe_roc_auc(
            prepared.token_labels,
            prepared.token_scores,
            sample_weight=token_weights,
        )
        response_auc = safe_roc_auc(
            prepared.response_labels,
            prepared.response_scores,
            sample_weight=cluster_weights,
        )
        if np.isfinite(token_auc):
            token_values.append(token_auc)
        if np.isfinite(response_auc):
            response_values.append(response_auc)
    token_low, token_high = _confidence_interval(token_values, confidence)
    response_low, response_high = _confidence_interval(response_values, confidence)
    return {
        "token_ci_low": token_low,
        "token_ci_high": token_high,
        "token_valid": len(token_values),
        "response_ci_low": response_low,
        "response_ci_high": response_high,
        "response_valid": len(response_values),
    }


def _paired_bootstrap(
    first: _PreparedClusters,
    second: _PreparedClusters,
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    cluster_count = len(first.clusters)
    if cluster_count != len(second.clusters):
        raise ValueError("Paired bootstrap inputs have different cluster counts")
    if cluster_count == 0 or n_bootstrap == 0:
        return _empty_bootstrap()
    rng = np.random.default_rng(seed)
    token_differences: list[float] = []
    response_differences: list[float] = []
    for _ in range(n_bootstrap):
        drawn = rng.integers(0, cluster_count, size=cluster_count)
        cluster_weights = np.bincount(drawn, minlength=cluster_count).astype(np.float64)
        first_token = safe_roc_auc(
            first.token_labels,
            first.token_scores,
            sample_weight=cluster_weights[first.token_cluster_index],
        )
        second_token = safe_roc_auc(
            second.token_labels,
            second.token_scores,
            sample_weight=cluster_weights[second.token_cluster_index],
        )
        first_response = safe_roc_auc(
            first.response_labels,
            first.response_scores,
            sample_weight=cluster_weights,
        )
        second_response = safe_roc_auc(
            second.response_labels,
            second.response_scores,
            sample_weight=cluster_weights,
        )
        if np.isfinite(first_token) and np.isfinite(second_token):
            token_differences.append(first_token - second_token)
        if np.isfinite(first_response) and np.isfinite(second_response):
            response_differences.append(first_response - second_response)
    token_low, token_high = _confidence_interval(token_differences, confidence)
    response_low, response_high = _confidence_interval(response_differences, confidence)
    return {
        "token_ci_low": token_low,
        "token_ci_high": token_high,
        "token_valid": len(token_differences),
        "response_ci_low": response_low,
        "response_ci_high": response_high,
        "response_valid": len(response_differences),
    }


def _bootstrap_macro_metrics(
    prepared_by_task: Sequence[_PreparedClusters],
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    if not prepared_by_task or any(
        len(prepared.clusters) == 0 for prepared in prepared_by_task
    ):
        return _empty_bootstrap()
    rng = np.random.default_rng(seed)
    token_values: list[float] = []
    response_values: list[float] = []
    for _ in range(n_bootstrap):
        task_token_auc: list[float] = []
        task_response_auc: list[float] = []
        for prepared in prepared_by_task:
            cluster_count = len(prepared.clusters)
            drawn = rng.integers(0, cluster_count, size=cluster_count)
            cluster_weights = np.bincount(
                drawn, minlength=cluster_count
            ).astype(np.float64)
            task_token_auc.append(
                safe_roc_auc(
                    prepared.token_labels,
                    prepared.token_scores,
                    sample_weight=cluster_weights[prepared.token_cluster_index],
                )
            )
            task_response_auc.append(
                safe_roc_auc(
                    prepared.response_labels,
                    prepared.response_scores,
                    sample_weight=cluster_weights,
                )
            )
        token_macro = _strict_mean(task_token_auc)
        response_macro = _strict_mean(task_response_auc)
        if np.isfinite(token_macro):
            token_values.append(token_macro)
        if np.isfinite(response_macro):
            response_values.append(response_macro)
    token_low, token_high = _confidence_interval(token_values, confidence)
    response_low, response_high = _confidence_interval(response_values, confidence)
    return {
        "token_ci_low": token_low,
        "token_ci_high": token_high,
        "token_valid": len(token_values),
        "response_ci_low": response_low,
        "response_ci_high": response_high,
        "response_valid": len(response_values),
    }


def _paired_macro_bootstrap(
    paired_by_task: Sequence[tuple[_PreparedClusters, _PreparedClusters]],
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    if not paired_by_task or any(
        len(first.clusters) == 0 or len(first.clusters) != len(second.clusters)
        for first, second in paired_by_task
    ):
        return _empty_bootstrap()
    rng = np.random.default_rng(seed)
    token_values: list[float] = []
    response_values: list[float] = []
    for _ in range(n_bootstrap):
        task_token_differences: list[float] = []
        task_response_differences: list[float] = []
        for first, second in paired_by_task:
            cluster_count = len(first.clusters)
            drawn = rng.integers(0, cluster_count, size=cluster_count)
            cluster_weights = np.bincount(
                drawn, minlength=cluster_count
            ).astype(np.float64)
            first_token = safe_roc_auc(
                first.token_labels,
                first.token_scores,
                sample_weight=cluster_weights[first.token_cluster_index],
            )
            second_token = safe_roc_auc(
                second.token_labels,
                second.token_scores,
                sample_weight=cluster_weights[second.token_cluster_index],
            )
            first_response = safe_roc_auc(
                first.response_labels,
                first.response_scores,
                sample_weight=cluster_weights,
            )
            second_response = safe_roc_auc(
                second.response_labels,
                second.response_scores,
                sample_weight=cluster_weights,
            )
            task_token_differences.append(first_token - second_token)
            task_response_differences.append(first_response - second_response)
        token_macro = _strict_mean(task_token_differences)
        response_macro = _strict_mean(task_response_differences)
        if np.isfinite(token_macro):
            token_values.append(token_macro)
        if np.isfinite(response_macro):
            response_values.append(response_macro)
    token_low, token_high = _confidence_interval(token_values, confidence)
    response_low, response_high = _confidence_interval(response_values, confidence)
    return {
        "token_ci_low": token_low,
        "token_ci_high": token_high,
        "token_valid": len(token_values),
        "response_ci_low": response_low,
        "response_ci_high": response_high,
        "response_valid": len(response_values),
    }


def _strict_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return float("nan")
    return float(array.mean())


def _confidence_interval(
    values: Iterable[float],
    confidence: float,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(array, [tail, 1.0 - tail])
    return float(low), float(high)


def _empty_bootstrap() -> dict[str, float | int]:
    return {
        "token_ci_low": float("nan"),
        "token_ci_high": float("nan"),
        "token_valid": 0,
        "response_ci_low": float("nan"),
        "response_ci_high": float("nan"),
        "response_valid": 0,
    }


def _scope_seed(seed: int, scope: str) -> int:
    digest = hashlib.sha256(f"{seed}:{scope}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _validate_bootstrap(n_bootstrap: int, confidence: float) -> None:
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")


def _validate_binary_column(series: pd.Series, name: str) -> None:
    if series.isna().any():
        raise ValueError(f"{name} contains missing values")
    unique = set(series.drop_duplicates().tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must be binary, got {sorted(unique)!r}")


def _validate_finite_scores(series: pd.Series, name: str) -> None:
    try:
        scores = series.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} contains non-finite values")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
