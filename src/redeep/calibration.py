"""Leakage-free calibration of token-level ReDeEP scores.

The public entry point, :func:`fit_calibration`, deliberately accepts separate
calibration-train and development data frames. Feature ranking and min/max
statistics are learned from calibration-train only; development labels are
used solely to choose ``K_heads``, ``K_layers``, and ``beta``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_TASKS = ("QA", "Data2txt", "Summary")
DEFAULT_BETAS = tuple(round(index / 10, 1) for index in range(1, 20))

_STANDARD_ECS = re.compile(r"^ecs_l(\d+)_h(\d+)$")
_LEGACY_ECS = re.compile(r"^ecs_whole_l(\d+)_h(\d+)$")
_STANDARD_PKS = re.compile(r"^pks_standard_l(\d+)$")
_LEGACY_PKS = re.compile(r"^pks_legacy_redeep_l(\d+)$")


@dataclass(frozen=True)
class AggregateMinMax:
    """Min/max parameters for one already-aggregated feature."""

    data_min: float
    data_max: float

    @classmethod
    def fit(cls, values: Sequence[float] | np.ndarray) -> AggregateMinMax:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            raise ValueError("Cannot fit a scaler on an empty array")
        if not np.isfinite(array).all():
            raise ValueError("Cannot fit a scaler with non-finite values")
        return cls(data_min=float(array.min()), data_max=float(array.max()))

    def transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("Cannot transform non-finite values")
        scale = self.data_max - self.data_min
        if scale == 0.0:
            return np.zeros_like(array, dtype=np.float64)
        # Match sklearn MinMaxScaler(clip=False): held-out values may leave [0, 1].
        return (array - self.data_min) / scale

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenCalibration:
    """All state needed to score held-out examples without refitting."""

    model_name: str
    mode: str
    selected_ecs_columns: tuple[str, ...]
    selected_pks_columns: tuple[str, ...]
    ecs_ranking: tuple[str, ...]
    pks_ranking: tuple[str, ...]
    ecs_rank_auc: dict[str, float | None]
    pks_rank_auc: dict[str, float | None]
    k_heads: int
    k_layers: int
    alpha: float
    beta: float
    ecs_scaler: AggregateMinMax
    pks_scaler: AggregateMinMax
    selection_metric: str
    selection_value: float
    dev_task_auc: dict[str, float | None]
    calibration_response_count: int
    dev_response_count: int
    schema_version: int = 1

    def score_frame(
        self,
        frame: pd.DataFrame,
        *,
        score_column: str = "redeep_score",
        copy: bool = True,
        include_components: bool = True,
    ) -> pd.DataFrame:
        """Apply frozen feature selection and scaling to a held-out frame."""

        return apply_calibration(
            frame,
            self,
            score_column=score_column,
            copy=copy,
            include_components=include_components,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FrozenCalibration:
        values = dict(payload)
        values["selected_ecs_columns"] = tuple(values["selected_ecs_columns"])
        values["selected_pks_columns"] = tuple(values["selected_pks_columns"])
        values["ecs_ranking"] = tuple(values["ecs_ranking"])
        values["pks_ranking"] = tuple(values["pks_ranking"])
        values["ecs_scaler"] = AggregateMinMax(**values["ecs_scaler"])
        values["pks_scaler"] = AggregateMinMax(**values["pks_scaler"])
        return cls(**values)


@dataclass(frozen=True)
class CalibrationSearchResult:
    """Frozen calibration plus the auditable development-grid trace."""

    frozen: FrozenCalibration
    grid_results: pd.DataFrame


def discover_feature_columns(
    frame: pd.DataFrame,
    mode: str = "standard",
) -> tuple[list[str], list[str]]:
    """Discover and numerically order ECS and PKS columns for a feature mode."""

    normalized_mode = _normalize_mode(mode)
    if normalized_mode == "standard":
        ecs_pattern = _STANDARD_ECS
        pks_pattern = _STANDARD_PKS
    else:
        ecs_pattern = _LEGACY_ECS
        pks_pattern = _LEGACY_PKS

    def matches(pattern: re.Pattern[str]) -> list[tuple[tuple[int, ...], str]]:
        found: list[tuple[tuple[int, ...], str]] = []
        for column in frame.columns:
            match = pattern.fullmatch(str(column))
            if match:
                found.append((tuple(int(value) for value in match.groups()), str(column)))
        return sorted(found, key=lambda item: (item[0], item[1]))

    ecs_columns = [column for _, column in matches(ecs_pattern)]
    pks_columns = [column for _, column in matches(pks_pattern)]
    if not ecs_columns:
        raise ValueError(f"No ECS columns found for mode={normalized_mode!r}")
    if not pks_columns:
        raise ValueError(f"No PKS columns found for mode={normalized_mode!r}")
    return ecs_columns, pks_columns


def rank_features(
    calibration_train: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    label_column: str = "token_label",
    inverse: bool = False,
) -> tuple[tuple[str, ...], dict[str, float | None]]:
    """Rank features by train-only token AUROC.

    ECS is expected to be *lower* on hallucinated tokens, so callers set
    ``inverse=True``. Features with undefined AUC are placed last, with column
    name as a deterministic tie breaker.
    """

    _require_columns(calibration_train, [label_column, *feature_columns])
    labels = _binary_labels(calibration_train[label_column], label_column)
    scores: dict[str, float | None] = {}
    for column in feature_columns:
        values = _finite_feature(calibration_train[column], column)
        auc = _safe_auc(labels, -values if inverse else values)
        scores[column] = None if np.isnan(auc) else float(auc)

    input_order = {column: index for index, column in enumerate(feature_columns)}
    ranking = tuple(
        sorted(
            feature_columns,
            key=lambda column: (
                -(scores[column] if scores[column] is not None else -np.inf),
                input_order[column],
                column,
            ),
        )
    )
    return ranking, scores


def search_calibration(
    calibration_train: pd.DataFrame,
    dev: pd.DataFrame,
    *,
    model_name: str,
    mode: str = "standard",
    ecs_columns: Sequence[str] | None = None,
    pks_columns: Sequence[str] | None = None,
    tasks: Sequence[str] = DEFAULT_TASKS,
    head_grid: Iterable[int] | None = None,
    layer_grid: Iterable[int] | None = None,
    beta_grid: Iterable[float] = DEFAULT_BETAS,
    alpha: float = 1.0,
    label_column: str = "token_label",
    task_column: str = "task",
    response_column: str = "response_id",
) -> CalibrationSearchResult:
    """Fit train-only rankings/scalers and select hyperparameters on dev."""

    normalized_mode = _normalize_mode(mode)
    task_names = tuple(str(task) for task in tasks)
    if not task_names or len(set(task_names)) != len(task_names):
        raise ValueError("tasks must be a non-empty sequence of unique names")
    _validate_partition(calibration_train, "calibration_train")
    _validate_partition(dev, "dev")
    _assert_disjoint_responses(
        calibration_train,
        dev,
        task_column=task_column,
        response_column=response_column,
    )
    _require_columns(calibration_train, [label_column, task_column, response_column])
    _require_columns(dev, [label_column, task_column, response_column])

    if ecs_columns is None or pks_columns is None:
        discovered_ecs, discovered_pks = discover_feature_columns(
            calibration_train, normalized_mode
        )
        ecs_columns = discovered_ecs if ecs_columns is None else list(ecs_columns)
        pks_columns = discovered_pks if pks_columns is None else list(pks_columns)
    else:
        ecs_columns = list(ecs_columns)
        pks_columns = list(pks_columns)

    _require_columns(calibration_train, [*ecs_columns, *pks_columns])
    _require_columns(dev, [*ecs_columns, *pks_columns])
    if not ecs_columns or not pks_columns:
        raise ValueError("At least one ECS and one PKS feature are required")

    ecs_ranking, ecs_rank_auc = rank_features(
        calibration_train, ecs_columns, label_column=label_column, inverse=True
    )
    pks_ranking, pks_rank_auc = rank_features(
        calibration_train, pks_columns, label_column=label_column, inverse=False
    )

    head_values = _validated_grid(
        head_grid, upper=min(32, len(ecs_ranking)), grid_name="head_grid"
    )
    layer_values = _validated_grid(
        layer_grid, upper=min(32, len(pks_ranking)), grid_name="layer_grid"
    )
    beta_values = tuple(float(value) for value in beta_grid)
    if not beta_values or not all(np.isfinite(beta_values)):
        raise ValueError("beta_grid must contain at least one finite value")
    if any(value <= 0 for value in beta_values):
        raise ValueError("beta_grid values must be positive")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")

    train_ecs = _numeric_matrix(calibration_train, ecs_ranking)
    dev_ecs = _numeric_matrix(dev, ecs_ranking)
    train_pks = _numeric_matrix(calibration_train, pks_ranking)
    dev_pks = _numeric_matrix(dev, pks_ranking)

    ecs_candidates = _scaled_prefix_candidates(train_ecs, dev_ecs, head_values)
    pks_candidates = _scaled_prefix_candidates(train_pks, dev_pks, layer_values)
    dev_labels = _binary_labels(dev[label_column], label_column)
    dev_tasks = dev[task_column].astype(str).to_numpy()

    records: list[dict[str, Any]] = []
    best_key: tuple[float, int, int, int, float] | None = None
    best_state: tuple[int, int, float, AggregateMinMax, AggregateMinMax] | None = None
    best_task_auc: dict[str, float | None] | None = None

    for k_heads in head_values:
        ecs_scaled, ecs_scaler = ecs_candidates[k_heads]
        for k_layers in layer_values:
            pks_scaled, pks_scaler = pks_candidates[k_layers]
            for beta in beta_values:
                scores = alpha * pks_scaled - beta * ecs_scaled
                task_auc, objective = _macro_task_auc(
                    dev_labels, scores, dev_tasks, tasks=task_names
                )
                record: dict[str, Any] = {
                    "k_heads": k_heads,
                    "k_layers": k_layers,
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "macro_task_token_auc": None
                    if np.isnan(objective)
                    else float(objective),
                }
                record.update({f"auc_{task}": task_auc[task] for task in task_names})
                records.append(record)

                if np.isnan(objective):
                    continue
                # Deterministic tie break: fewer total features, then fewer
                # heads/layers, then lower beta.
                candidate_key = (
                    float(objective),
                    -(k_heads + k_layers),
                    -k_heads,
                    -k_layers,
                    -float(beta),
                )
                if best_key is None or candidate_key > best_key:
                    best_key = candidate_key
                    best_state = (
                        k_heads,
                        k_layers,
                        float(beta),
                        ecs_scaler,
                        pks_scaler,
                    )
                    best_task_auc = task_auc

    if best_state is None or best_key is None or best_task_auc is None:
        raise ValueError(
            "At least one required development task is missing or single-class; "
            "the task-macro calibration objective is undefined"
        )

    k_heads, k_layers, beta, ecs_scaler, pks_scaler = best_state
    frozen = FrozenCalibration(
        model_name=model_name,
        mode=normalized_mode,
        selected_ecs_columns=ecs_ranking[:k_heads],
        selected_pks_columns=pks_ranking[:k_layers],
        ecs_ranking=ecs_ranking,
        pks_ranking=pks_ranking,
        ecs_rank_auc=ecs_rank_auc,
        pks_rank_auc=pks_rank_auc,
        k_heads=k_heads,
        k_layers=k_layers,
        alpha=float(alpha),
        beta=beta,
        ecs_scaler=ecs_scaler,
        pks_scaler=pks_scaler,
        selection_metric="macro_task_token_micro_roc_auc",
        selection_value=float(best_key[0]),
        dev_task_auc=best_task_auc,
        calibration_response_count=_response_count(
            calibration_train, task_column, response_column
        ),
        dev_response_count=_response_count(dev, task_column, response_column),
    )
    return CalibrationSearchResult(frozen=frozen, grid_results=pd.DataFrame.from_records(records))


def fit_calibration(
    calibration_train: pd.DataFrame,
    dev: pd.DataFrame,
    **kwargs: Any,
) -> FrozenCalibration:
    """Return only the selected, frozen calibration state."""

    return search_calibration(calibration_train, dev, **kwargs).frozen


def fit_calibration_from_splits(
    frame: pd.DataFrame,
    *,
    train_value: str = "calibration_train",
    dev_value: str = "dev",
    split_column: str = "eval_split",
    **kwargs: Any,
) -> FrozenCalibration:
    """Convenience wrapper for a feature table carrying ``eval_split``."""

    _require_columns(frame, [split_column])
    calibration_train = frame.loc[frame[split_column] == train_value].copy()
    dev = frame.loc[frame[split_column] == dev_value].copy()
    if calibration_train.empty:
        raise ValueError(f"No rows found for {split_column}={train_value!r}")
    if dev.empty:
        raise ValueError(f"No rows found for {split_column}={dev_value!r}")
    return fit_calibration(calibration_train, dev, **kwargs)


def apply_calibration(
    frame: pd.DataFrame,
    calibration: FrozenCalibration,
    *,
    score_column: str = "redeep_score",
    copy: bool = True,
    include_components: bool = True,
) -> pd.DataFrame:
    """Score a frame using only frozen train/dev calibration state."""

    selected = [
        *calibration.selected_ecs_columns,
        *calibration.selected_pks_columns,
    ]
    _require_columns(frame, selected)
    output = frame.copy() if copy else frame
    ecs_sum = _numeric_matrix(output, calibration.selected_ecs_columns).sum(axis=1)
    pks_sum = _numeric_matrix(output, calibration.selected_pks_columns).sum(axis=1)
    ecs_scaled = calibration.ecs_scaler.transform(ecs_sum)
    pks_scaled = calibration.pks_scaler.transform(pks_sum)
    output[score_column] = (
        calibration.alpha * pks_scaled - calibration.beta * ecs_scaled
    )
    if include_components:
        output["redeep_ecs_sum"] = ecs_sum
        output["redeep_pks_sum"] = pks_sum
        output["redeep_ecs_scaled"] = ecs_scaled
        output["redeep_pks_scaled"] = pks_scaled
    return output


def _normalize_mode(mode: str) -> str:
    aliases = {
        "paper": "standard",
        "standard": "standard",
        "legacy": "legacy_redeep",
        "legacy_redeep": "legacy_redeep",
    }
    try:
        return aliases[mode.lower()]
    except KeyError as error:
        raise ValueError(
            f"Unsupported feature mode {mode!r}; expected standard or legacy_redeep"
        ) from error


def _validated_grid(
    values: Iterable[int] | None,
    *,
    upper: int,
    grid_name: str,
) -> tuple[int, ...]:
    if upper < 1:
        raise ValueError(f"No candidates available for {grid_name}")
    result = tuple(range(1, upper + 1)) if values is None else tuple(int(v) for v in values)
    if not result:
        raise ValueError(f"{grid_name} cannot be empty")
    if any(value < 1 or value > upper for value in result):
        raise ValueError(f"{grid_name} values must be within [1, {upper}]")
    return tuple(dict.fromkeys(result))


def _scaled_prefix_candidates(
    train_matrix: np.ndarray,
    dev_matrix: np.ndarray,
    grid: Sequence[int],
) -> dict[int, tuple[np.ndarray, AggregateMinMax]]:
    train_cumulative = train_matrix.cumsum(axis=1)
    dev_cumulative = dev_matrix.cumsum(axis=1)
    candidates: dict[int, tuple[np.ndarray, AggregateMinMax]] = {}
    for size in grid:
        train_sum = train_cumulative[:, size - 1]
        dev_sum = dev_cumulative[:, size - 1]
        scaler = AggregateMinMax.fit(train_sum)
        candidates[size] = (scaler.transform(dev_sum), scaler)
    return candidates


def _macro_task_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    task_values: np.ndarray,
    *,
    tasks: Sequence[str],
) -> tuple[dict[str, float | None], float]:
    task_auc: dict[str, float | None] = {}
    valid_auc: list[float] = []
    for task in tasks:
        mask = task_values == str(task)
        auc = _safe_auc(labels[mask], scores[mask])
        if np.isnan(auc):
            task_auc[str(task)] = None
        else:
            task_auc[str(task)] = float(auc)
            valid_auc.append(float(auc))
    objective = (
        float(np.mean(valid_auc))
        if len(valid_auc) == len(tasks)
        else float("nan")
    )
    return task_auc, objective


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _binary_labels(series: pd.Series, name: str) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f"{name} contains missing values")
    values = series.to_numpy()
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must be binary, got {sorted(unique)!r}")
    return values.astype(np.int8, copy=False)


def _finite_feature(series: pd.Series, name: str) -> np.ndarray:
    try:
        values = series.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Feature {name!r} is not numeric") from error
    if not np.isfinite(values).all():
        raise ValueError(f"Feature {name!r} contains non-finite values")
    return values


def _numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    try:
        matrix = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Feature columns must be numeric") from error
    if not np.isfinite(matrix).all():
        raise ValueError("Feature columns contain non-finite values")
    return matrix


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_partition(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} is empty")
    for source_split_column in ("original_split", "source_split", "split"):
        if source_split_column in frame:
            bad = frame[source_split_column].astype(str).str.lower().eq("test")
            if bad.any():
                raise ValueError(f"{name} contains original test rows")
    if "eval_split" in frame:
        bad = frame["eval_split"].astype(str).str.lower().eq("test")
        if bad.any():
            raise ValueError(f"{name} contains eval_split='test' rows")


def _assert_disjoint_responses(
    calibration_train: pd.DataFrame,
    dev: pd.DataFrame,
    *,
    task_column: str,
    response_column: str,
) -> None:
    _require_columns(calibration_train, [task_column, response_column])
    _require_columns(dev, [task_column, response_column])
    train_keys = set(
        zip(
            calibration_train[task_column].astype(str),
            calibration_train[response_column].astype(str),
            strict=True,
        )
    )
    dev_keys = set(
        zip(
            dev[task_column].astype(str),
            dev[response_column].astype(str),
            strict=True,
        )
    )
    overlap = train_keys.intersection(dev_keys)
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(f"Calibration-train/dev response leakage detected: {preview}")


def _response_count(frame: pd.DataFrame, task_column: str, response_column: str) -> int:
    return int(frame.loc[:, [task_column, response_column]].drop_duplicates().shape[0])
