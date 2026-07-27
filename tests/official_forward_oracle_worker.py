"""Subprocess worker for released-vs-modern ReDeEP forward parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as functional
from transformers import LlamaConfig, LlamaForCausalLM


def config() -> LlamaConfig:
    value = LlamaConfig(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
    )
    value._attn_implementation = "eager"
    return value


def inputs() -> tuple[torch.Tensor, int, list[int], list[tuple[int, int]]]:
    input_ids = torch.tensor(
        [[(index * 7 + 3) % 97 + 1 for index in range(28)]],
        dtype=torch.long,
    )
    prefix_length = 20
    score_positions = list(range(prefix_length - 1, input_ids.shape[-1] - 1))
    candidate_heads = [(0, 0), (1, 3)]
    return input_ids, prefix_length, score_positions, candidate_heads


def reverse_jsd(post_logits: torch.Tensor, pre_logits: torch.Tensor) -> torch.Tensor:
    post_probability = functional.softmax(post_logits, dim=-1)
    pre_probability = functional.softmax(pre_logits, dim=-1)
    midpoint = 0.5 * (post_probability + pre_probability)
    post_term = functional.kl_div(
        functional.log_softmax(post_logits, dim=-1),
        midpoint,
        reduction="none",
    ).mean(-1)
    pre_term = functional.kl_div(
        functional.log_softmax(pre_logits, dim=-1),
        midpoint,
        reduction="none",
    ).mean(-1)
    return 0.5 * (post_term + pre_term) * 1_000_000.0


def released_features(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    prefix_length: int,
    score_positions: list[int],
    candidate_heads: list[tuple[int, int]],
) -> dict[str, object]:
    with torch.inference_mode():
        logits_by_layer, outputs = model(
            input_ids=input_ids,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            knowledge_layers=list(range(model.config.num_hidden_layers)),
        )
    final_hidden = outputs.hidden_states[-1][0]
    external = []
    top_indices = []
    top_count = int(prefix_length * 0.1)
    for position in score_positions:
        token_scores = []
        token_indices = []
        for layer_index, head_index in candidate_heads:
            attention = outputs.attentions[layer_index][
                0, head_index, position, :prefix_length
            ]
            indices = torch.argsort(attention, descending=True)[:top_count]
            attended_hidden = final_hidden.index_select(0, indices).mean(0)
            token_scores.append(
                functional.cosine_similarity(
                    attended_hidden,
                    final_hidden[position],
                    dim=0,
                ).item()
            )
            token_indices.append(indices.tolist())
        external.append(token_scores)
        top_indices.append(token_indices)

    parametric = []
    for position in score_positions:
        parametric.append(
            [
                reverse_jsd(
                    logits_by_layer[layer_index][0][0, position],
                    logits_by_layer[layer_index][1][0, position],
                ).item()
                for layer_index in range(model.config.num_hidden_layers)
            ]
        )
    return {
        "external": external,
        "parametric": parametric,
        "top_indices": top_indices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("released", "modern"), required=True)
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(1729)
    model = LlamaForCausalLM(config()).eval()
    if args.mode == "released":
        torch.save(model.state_dict(), args.state_dict)
        input_ids, prefix_length, score_positions, candidate_heads = inputs()
        result = released_features(
            model,
            input_ids,
            prefix_length,
            score_positions,
            candidate_heads,
        )
    else:
        model.load_state_dict(
            torch.load(
                args.state_dict,
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        from redeep_token.scoring import ReDeEPFeatureExtractor

        input_ids, prefix_length, score_positions, candidate_heads = inputs()
        extractor = ReDeEPFeatureExtractor(
            model,
            candidate_heads=candidate_heads,
            top_fraction=0.1,
            logit_chunk_size=2,
            cosine_chunk_size=2,
        )
        result = extractor.extract(
            input_ids=input_ids[0].tolist(),
            prefix_length=prefix_length,
            score_positions=score_positions,
        )

    args.output.write_text(
        json.dumps(result, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
