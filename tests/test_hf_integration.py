from __future__ import annotations

import pytest


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_real_transformers_decoder_adapter_and_features(family: str) -> None:
    """Exercise the exact Transformers classes used by the remote run."""

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from redeep.features import extract_token_features
    from redeep.models import DecoderModelAdapter, eager_attention_parity

    common = {
        "vocab_size": 67,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    if family == "llama":
        config = transformers.LlamaConfig(**common)
        model_class = transformers.LlamaForCausalLM
    else:
        if not hasattr(transformers, "Qwen3Config"):
            pytest.skip("Installed Transformers does not include Qwen3")
        config = transformers.Qwen3Config(**common)
        model_class = transformers.Qwen3ForCausalLM
    config._attn_implementation = "eager"
    torch.manual_seed(123)
    model = model_class(config).eval()
    adapter = DecoderModelAdapter.from_model(model)
    input_ids = torch.tensor([[1, 3, 4, 5, 6, 7]])
    attention_mask = torch.ones_like(input_ids)

    parity = eager_attention_parity(
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        layer_index=1,
        head_indices=(0, 3),
        query_positions=(3, 5),
        key_positions=(0, 1, 2, 3, 4, 5),
    )
    assert parity.allclose, parity

    batch = extract_token_features(
        adapter,
        input_ids,
        attention_mask,
        predictor_positions=(3, 4, 5),
        context_positions=(0, 1, 2),
        copying_heads=((0, 0), (1, 3)),
        pks_layers=(0, 1),
        pks_modes=("standard", "legacy_redeep"),
        top_fraction=0.5,
        vocab_chunk_size=100,
        token_chunk_size=2,
        whole_prefix_positions=(0, 1, 2, 3),
    )
    assert set(batch.columns) == {
        "ecs_l0_h0",
        "ecs_l1_h3",
        "ecs_whole_l0_h0",
        "ecs_whole_l1_h3",
        "pks_standard_l0",
        "pks_standard_l1",
        "pks_legacy_redeep_l0",
        "pks_legacy_redeep_l1",
    }
    assert all(tuple(values.shape) == (1, 3) for values in batch.columns.values())
    assert all(torch.isfinite(values).all() for values in batch.columns.values())
