from __future__ import annotations

# ruff: noqa: E402, I001

import math

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from redeep.copy_heads import (
    discover_copy_heads,
    estimate_sampled_gershgorin,
    exact_low_rank_trace,
)
from redeep.features import (
    compute_ecs,
    extract_token_features,
    js_divergence_from_logits,
    streaming_js_divergence,
)
from redeep.models import DecoderModelAdapter
from test_models import TinyCausalLM


def test_standard_jsd_is_symmetric_bounded_and_zero_on_identity():
    first = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 1.0, 4.0]])
    second = torch.tensor([[-0.5, 2.5, 1.0], [2.0, -1.0, 0.0]])
    forward = js_divergence_from_logits(first, second, mode="standard")
    reverse = js_divergence_from_logits(second, first, mode="standard")
    torch.testing.assert_close(forward, reverse)
    assert torch.all(forward >= 0)
    assert torch.all(forward <= math.log(2.0) + 1e-6)
    torch.testing.assert_close(
        js_divergence_from_logits(first, first, mode="standard"),
        torch.zeros(2),
        atol=1e-7,
        rtol=0,
    )


def test_legacy_jsd_matches_released_reverse_kl_scale():
    first = torch.tensor([[2.0, -1.0, 0.5]])
    second = torch.tensor([[-0.5, 2.5, 1.0]])
    log_first = F.log_softmax(first, dim=-1)
    log_second = F.log_softmax(second, dim=-1)
    mixture = 0.5 * (log_first.exp() + log_second.exp())
    expected = (
        0.5
        * (
            F.kl_div(log_first, mixture, reduction="none").mean(dim=-1)
            + F.kl_div(log_second, mixture, reduction="none").mean(dim=-1)
        )
        * 1_000_000
    )
    actual = js_divergence_from_logits(first, second, mode="legacy_redeep")
    torch.testing.assert_close(actual, expected)


def test_streaming_jsd_matches_full_logits():
    torch.manual_seed(11)
    adapter = DecoderModelAdapter.from_model(TinyCausalLM())
    before = torch.randn(1, 5, adapter.hidden_size)
    after = before + 0.2 * torch.randn_like(before)
    normalized_before = adapter.normalize_for_logit_lens(before)
    normalized_after = adapter.normalize_for_logit_lens(after)
    before_logits = adapter.lm_head(normalized_before)
    after_logits = adapter.lm_head(normalized_after)

    streamed = streaming_js_divergence(
        adapter,
        before,
        after,
        modes=("standard", "legacy_redeep"),
        vocab_chunk_size=3,
        token_chunk_size=2,
    )
    for mode in ("standard", "legacy_redeep"):
        expected = js_divergence_from_logits(
            before_logits,
            after_logits,
            mode=mode,
        )
        if mode == "standard":
            torch.testing.assert_close(streamed[mode], expected, atol=2e-5, rtol=2e-5)
        else:
            # The legacy mean is amplified by 1e6, so harmless blockwise
            # floating-point summation differences are amplified as well.
            torch.testing.assert_close(streamed[mode], expected, atol=1e-2, rtol=1e-4)


def test_full_vocab_fast_path_matches_reference():
    torch.manual_seed(13)
    adapter = DecoderModelAdapter.from_model(TinyCausalLM())
    before = torch.randn(1, 3, adapter.hidden_size)
    after = before + 0.1 * torch.randn_like(before)
    expected_before = adapter.lm_head(adapter.normalize_for_logit_lens(before))
    expected_after = adapter.lm_head(adapter.normalize_for_logit_lens(after))
    actual = streaming_js_divergence(
        adapter,
        before,
        after,
        modes=("standard", "legacy_redeep"),
        vocab_chunk_size=adapter.vocab_size,
        token_chunk_size=2,
    )
    for mode in actual:
        expected = js_divergence_from_logits(
            expected_before,
            expected_after,
            mode=mode,
        )
        if mode == "standard":
            torch.testing.assert_close(actual[mode], expected)
        else:
            torch.testing.assert_close(actual[mode], expected, atol=3e-3, rtol=1e-4)


def test_ecs_uses_per_query_valid_key_count_and_floor_rounding():
    attention = torch.tensor(
        [[[[9.0, 1000.0, 999.0], [3.0, 2.0, 1.0]]]]
    )
    context = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    query = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    valid = torch.tensor([[[True, False, False], [True, True, True]]])
    score = compute_ecs(
        attention,
        query,
        context,
        valid_key_mask=valid,
        top_fraction=0.5,
    )
    assert score.shape == (1, 1, 2)
    torch.testing.assert_close(score[0, 0, 0], torch.tensor(1.0))
    expected_second = F.cosine_similarity(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 0.0]]),
    )[0]
    torch.testing.assert_close(score[0, 0, 1], expected_second)


def test_exact_trace_matches_materialized_vocabulary_circuit():
    torch.manual_seed(19)
    vocabulary, hidden, head_dim = 7, 5, 2
    embedding = torch.randn(vocabulary, hidden)
    unembedding = torch.randn(vocabulary, hidden)
    value = torch.randn(head_dim, hidden)
    output = torch.randn(hidden, head_dim)
    gram = unembedding.T @ embedding
    expected = torch.trace(
        embedding @ value.T @ output.T @ unembedding.T
    )
    actual = exact_low_rank_trace(gram, value, output)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_full_sample_gershgorin_matches_materialized_radii():
    torch.manual_seed(23)
    vocabulary, hidden, head_dim = 6, 4, 2
    embedding = torch.randn(vocabulary, hidden)
    unembedding = torch.randn(vocabulary, hidden)
    value = torch.randn(head_dim, hidden)
    output = torch.randn(hidden, head_dim)
    matrix = embedding @ value.T @ output.T @ unembedding.T
    diagonal = matrix.diagonal()
    radii = matrix.abs().sum(dim=-1) - diagonal.abs()
    estimate = estimate_sampled_gershgorin(
        embedding,
        unembedding,
        value,
        output,
        vocabulary_size=vocabulary,
        row_block_size=2,
    )
    assert estimate.mean_estimated_radius == pytest.approx(
        radii.mean().item(),
        rel=1e-5,
    )
    assert estimate.maximum_estimated_radius == pytest.approx(
        radii.max().item(),
        rel=1e-5,
    )


def test_copy_head_discovery_records_gqa_and_approximation_metadata():
    torch.manual_seed(29)
    adapter = DecoderModelAdapter.from_model(TinyCausalLM())
    discovery = discover_copy_heads(
        adapter,
        top_k=3,
        vocab_block_size=4,
        gershgorin_sample_size=adapter.vocab_size,
        gershgorin_block_size=3,
        seed=5,
        compute_device="cpu",
    )
    assert len(discovery.records) == adapter.num_layers * adapter.num_attention_heads
    assert len(discovery.top_heads) == 3
    by_layer_head = sorted(discovery.records, key=lambda row: (row.layer, row.head))
    assert [row.kv_head for row in by_layer_head[:4]] == [0, 0, 1, 1]
    assert discovery.metadata["gershgorin_exact"] is True
    assert len(discovery.metadata["sample_indices"]) == adapter.vocab_size


def test_end_to_end_tiny_feature_columns_include_fixed_prompt_prefix():
    torch.manual_seed(31)
    adapter = DecoderModelAdapter.from_model(TinyCausalLM(model_type="qwen3", qk_norm=True))
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    features = extract_token_features(
        adapter,
        input_ids,
        torch.ones_like(input_ids),
        predictor_positions=(2, 3, 4),
        context_positions=(0, 1),
        copying_heads=((0, 0),),
        pks_layers=(0,),
        pks_modes=("standard", "legacy_redeep"),
        top_fraction=0.5,
        vocab_chunk_size=4,
        token_chunk_size=2,
        whole_prefix_positions=(0, 1, 2),
    )
    assert set(features.columns) == {
        "ecs_l0_h0",
        "ecs_whole_l0_h0",
        "pks_standard_l0",
        "pks_legacy_redeep_l0",
    }
    assert all(value.shape == (1, 3) for value in features.columns.values())
    assert all(torch.isfinite(value).all() for value in features.columns.values())

    with torch.inference_mode():
        reference = adapter.decoder(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )
    predictor_index = torch.tensor([2, 3, 4])
    context_index = torch.tensor([0, 1])
    direct_context_attention = (
        reference.attentions[0][:, 0:1]
        .index_select(2, predictor_index)
        .index_select(3, context_index)
    )
    direct_context_ecs = compute_ecs(
        direct_context_attention,
        reference.last_hidden_state.index_select(1, predictor_index),
        reference.last_hidden_state.index_select(1, context_index),
        top_fraction=0.5,
    )
    torch.testing.assert_close(
        features.columns["ecs_l0_h0"],
        direct_context_ecs[:, 0],
        atol=1e-6,
        rtol=1e-5,
    )

    whole_index = torch.arange(3)
    direct_whole_attention = (
        reference.attentions[0][:, 0:1]
        .index_select(2, predictor_index)
        .index_select(3, whole_index)
    )
    direct_whole_ecs = compute_ecs(
        direct_whole_attention,
        reference.last_hidden_state.index_select(1, predictor_index),
        reference.last_hidden_state.index_select(1, whole_index),
        top_fraction=0.5,
    )
    torch.testing.assert_close(
        features.columns["ecs_whole_l0_h0"],
        direct_whole_ecs[:, 0],
        atol=1e-6,
        rtol=1e-5,
    )
