from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from redeep.calibration import (
    FrozenCalibration,
    discover_feature_columns,
    fit_calibration,
    fit_calibration_from_splits,
    search_calibration,
)

TASKS = ("QA", "Data2txt", "Summary")


def _partition(prefix: str, eval_split: str, *, dev_outlier: bool = False) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for label in (0, 1, 0, 1):
            value = float(label)
            pks_good = 100.0 if dev_outlier and label == 1 else value
            rows.append(
                {
                    "response_id": f"{prefix}-{task}-{len(rows)}",
                    "source_id": f"source-{len(rows)}",
                    "task": task,
                    "original_split": "train",
                    "eval_split": eval_split,
                    "quality": "good",
                    "token_index": 0,
                    "token_label": label,
                    "response_label": label,
                    "ecs_l0_h0": 1.0 - value,
                    "ecs_l0_h1": value,
                    "pks_standard_l0": pks_good,
                    "pks_standard_l1": 1.0 - value,
                    "ecs_whole_l0_h0": 1.0 - value,
                    "pks_legacy_redeep_l0": value,
                }
            )
    return pd.DataFrame(rows)


def test_train_only_ranking_scaling_and_dev_selection() -> None:
    train = _partition("train", "calibration_train")
    dev = _partition("dev", "dev", dev_outlier=True)

    result = search_calibration(
        train,
        dev,
        model_name="synthetic",
        head_grid=(1, 2),
        layer_grid=(1, 2),
        beta_grid=(0.1, 1.0),
    )
    frozen = result.frozen

    assert frozen.ecs_ranking[0] == "ecs_l0_h0"
    assert frozen.pks_ranking[0] == "pks_standard_l0"
    assert frozen.k_heads == 1
    assert frozen.k_layers == 1
    assert frozen.beta == 0.1
    assert frozen.selection_value == pytest.approx(1.0)
    # The dev value of 100 must not leak into train-only min/max fitting.
    assert frozen.ecs_scaler.data_min == 0.0
    assert frozen.ecs_scaler.data_max == 1.0
    assert frozen.pks_scaler.data_min == 0.0
    assert frozen.pks_scaler.data_max == 1.0

    scored = frozen.score_frame(dev)
    assert scored.loc[scored["token_label"] == 1, "redeep_pks_scaled"].max() == 100.0
    assert np.isfinite(scored["redeep_score"]).all()
    assert len(result.grid_results) == 2 * 2 * 2


def test_fit_from_splits_and_frozen_round_trip() -> None:
    train = _partition("train", "calibration_train")
    dev = _partition("dev", "dev")
    combined = pd.concat([train, dev], ignore_index=True)
    frozen = fit_calibration_from_splits(
        combined,
        model_name="synthetic",
        head_grid=(1,),
        layer_grid=(1,),
        beta_grid=(0.4,),
    )

    restored = FrozenCalibration.from_dict(frozen.to_dict())
    pd.testing.assert_series_equal(
        frozen.score_frame(dev)["redeep_score"],
        restored.score_frame(dev)["redeep_score"],
    )


def test_standard_and_legacy_column_discovery_are_separate() -> None:
    frame = _partition("train", "calibration_train")
    standard_ecs, standard_pks = discover_feature_columns(frame, mode="standard")
    legacy_ecs, legacy_pks = discover_feature_columns(frame, mode="legacy_redeep")

    assert standard_ecs == ["ecs_l0_h0", "ecs_l0_h1"]
    assert standard_pks == ["pks_standard_l0", "pks_standard_l1"]
    assert legacy_ecs == ["ecs_whole_l0_h0"]
    assert legacy_pks == ["pks_legacy_redeep_l0"]


def test_response_overlap_is_rejected() -> None:
    train = _partition("same", "calibration_train")
    dev = train.copy()
    dev["eval_split"] = "dev"
    with pytest.raises(ValueError, match="leakage"):
        fit_calibration(
            train,
            dev,
            model_name="synthetic",
            head_grid=(1,),
            layer_grid=(1,),
            beta_grid=(0.1,),
        )


def test_original_test_rows_cannot_enter_calibration() -> None:
    train = _partition("train", "calibration_train")
    train.loc[0, "original_split"] = "test"
    dev = _partition("dev", "dev")
    with pytest.raises(ValueError, match="original test"):
        fit_calibration(
            train,
            dev,
            model_name="synthetic",
            head_grid=(1,),
            layer_grid=(1,),
            beta_grid=(0.1,),
        )


def test_all_single_class_dev_tasks_raise_clear_error() -> None:
    train = _partition("train", "calibration_train")
    dev = _partition("dev", "dev")
    dev["token_label"] = 0
    with pytest.raises(ValueError, match="single-class"):
        fit_calibration(
            train,
            dev,
            model_name="synthetic",
            head_grid=(1,),
            layer_grid=(1,),
            beta_grid=(0.1,),
        )


def test_missing_one_dev_task_cannot_be_silently_ignored() -> None:
    train = _partition("train", "calibration_train")
    dev = _partition("dev", "dev")
    dev = dev.loc[dev["task"] != "Summary"].copy()
    with pytest.raises(ValueError, match="missing or single-class"):
        fit_calibration(
            train,
            dev,
            model_name="synthetic",
            head_grid=(1,),
            layer_grid=(1,),
            beta_grid=(0.1,),
        )
