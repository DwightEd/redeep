from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from redeep.evaluation import compare_models_paired, evaluate_model, safe_roc_auc

TASKS = ("QA", "Data2txt", "Summary")


def _perfect_frame(*, reverse: bool = False, one_token: bool = False) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for response_index, response_label in enumerate((0, 1, 0, 1)):
            token_labels = (response_label,) if one_token else (
                (0, 0) if response_label == 0 else (0, 1)
            )
            for token_index, token_label in enumerate(token_labels):
                score = float(token_label)
                if reverse:
                    score = -score
                rows.append(
                    {
                        "response_id": f"{task}-{response_index}",
                        "source_id": f"source-{task}-{response_index}",
                        "task": task,
                        "split": "test",
                        "quality": "good",
                        "token_index": token_index,
                        "token_label": token_label,
                        "response_label": response_label,
                        "redeep_score": score,
                    }
                )
    return pd.DataFrame(rows)


def test_evaluate_model_reports_task_and_overall_auc_counts_and_ci() -> None:
    frame = _perfect_frame()
    report = evaluate_model(
        frame,
        model_name="model-a",
        mode="standard",
        n_bootstrap=100,
        seed=7,
    )

    assert report["task"].tolist() == [*TASKS, "Overall", "TaskMacro"]
    assert np.allclose(report["token_micro_roc_auc"], 1.0)
    assert np.allclose(report["paper_response_mean_token_roc_auc"], 1.0)
    qa = report.loc[report["task"] == "QA"].iloc[0]
    assert qa["n_responses"] == 4
    assert qa["n_tokens"] == 8
    assert qa["n_positive_tokens"] == 2
    assert qa["n_negative_tokens"] == 6
    assert qa["n_positive_responses"] == 2
    assert qa["token_auc_ci_low"] == pytest.approx(1.0)
    assert qa["token_auc_ci_high"] == pytest.approx(1.0)
    assert 0 < qa["token_bootstrap_valid"] <= 100


def test_bootstrap_is_deterministic() -> None:
    frame = _perfect_frame()
    first = evaluate_model(
        frame,
        model_name="model-a",
        mode="standard",
        n_bootstrap=50,
        seed=123,
    )
    second = evaluate_model(
        frame,
        model_name="model-a",
        mode="standard",
        n_bootstrap=50,
        seed=123,
    )
    pd.testing.assert_frame_equal(first, second)


def test_single_class_task_returns_nan_without_failing_other_tasks() -> None:
    frame = _perfect_frame()
    qa_mask = frame["task"] == "QA"
    frame.loc[qa_mask, ["token_label", "response_label"]] = 0
    report = evaluate_model(
        frame,
        model_name="model-a",
        mode="standard",
        n_bootstrap=20,
    )

    qa = report.loc[report["task"] == "QA"].iloc[0]
    assert np.isnan(qa["token_micro_roc_auc"])
    assert np.isnan(qa["paper_response_mean_token_roc_auc"])
    assert qa["token_bootstrap_valid"] == 0
    assert report.loc[report["task"] == "Summary", "token_micro_roc_auc"].item() == 1.0
    macro = report.loc[report["task"] == "TaskMacro"].iloc[0]
    assert np.isnan(macro["token_micro_roc_auc"])
    assert macro["token_bootstrap_valid"] == 0


def test_task_macro_is_the_mean_of_task_statistics() -> None:
    frame = _perfect_frame()
    summary = frame["task"] == "Summary"
    frame.loc[summary, "redeep_score"] *= -1
    report = evaluate_model(
        frame,
        model_name="model-a",
        mode="standard",
        n_bootstrap=0,
    )
    task_values = report.loc[
        report["task"].isin(TASKS), "token_micro_roc_auc"
    ].to_numpy()
    macro = report.loc[
        report["task"] == "TaskMacro", "token_micro_roc_auc"
    ].item()
    assert macro == pytest.approx(task_values.mean())


def test_paired_bootstrap_pairs_responses_despite_different_token_counts() -> None:
    first = _perfect_frame()
    second = _perfect_frame(reverse=True, one_token=True)
    comparison = compare_models_paired(
        first,
        second,
        first_model="good",
        second_model="bad",
        n_bootstrap=100,
        seed=9,
    )

    assert comparison["task"].tolist() == [*TASKS, "Overall", "TaskMacro"]
    assert np.allclose(comparison["token_auc_difference"], 1.0)
    assert np.allclose(comparison["response_auc_difference"], 1.0)
    assert np.allclose(comparison["token_difference_ci_low"], 1.0)
    assert np.allclose(comparison["token_difference_ci_high"], 1.0)
    assert (comparison["n_common_responses"] > 0).all()


def test_response_label_mismatch_between_models_is_rejected() -> None:
    first = _perfect_frame()
    second = _perfect_frame()
    second.loc[second["response_id"] == "QA-0", "response_label"] = 1
    with pytest.raises(ValueError, match="mismatch"):
        compare_models_paired(
            first,
            second,
            first_model="a",
            second_model="b",
            n_bootstrap=1,
        )


def test_safe_auc_handles_empty_and_single_class() -> None:
    assert np.isnan(safe_roc_auc([], []))
    assert np.isnan(safe_roc_auc([0, 0], [0.1, 0.2]))
