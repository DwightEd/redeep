"""Memory-bounded ReDeEP token features.

This module implements the paper JSD, the released code's legacy reverse-KL
variant, External Context Scores (ECS), and a one-forward-pass extraction
entrypoint.  Vocabulary logits are streamed in blocks and are never retained
for every layer/token at once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from redeep.models import DecoderModelAdapter

JSD_STANDARD = "standard"
JSD_LEGACY_REDEEP = "legacy_redeep"
JSD_MODES = frozenset({JSD_STANDARD, JSD_LEGACY_REDEEP})
LEGACY_SCALE = 1_000_000.0


def pks_feature_name(mode: str, layer: int) -> str:
    _validate_js_modes((mode,))
    return f"pks_{mode}_l{int(layer)}"


def ecs_feature_name(layer: int, head: int, *, whole_prefix: bool = False) -> str:
    prefix = "ecs_whole" if whole_prefix else "ecs"
    return f"{prefix}_l{int(layer)}_h{int(head)}"


def _validate_js_modes(modes: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(mode) for mode in modes)
    if not result:
        raise ValueError("At least one JSD mode is required")
    if len(set(result)) != len(result):
        raise ValueError("JSD modes cannot contain duplicates")
    invalid = sorted(set(result) - JSD_MODES)
    if invalid:
        raise ValueError(f"Unknown JSD modes {invalid}; expected {sorted(JSD_MODES)}")
    return result


def js_divergence_from_logits(
    before_logits: Tensor,
    after_logits: Tensor,
    *,
    mode: str = JSD_STANDARD,
) -> Tensor:
    """Calculate JSD directly from logits for reference/testing.

    ``standard`` is the paper definition
    ``0.5 KL(P || M) + 0.5 KL(Q || M)``.  ``legacy_redeep`` exactly captures
    the released implementation's direction and scale:
    ``mean_vocab(0.5 KL(M || P) + 0.5 KL(M || Q)) * 1e6``.
    """

    _validate_js_modes((mode,))
    if before_logits.shape != after_logits.shape:
        raise ValueError("before_logits and after_logits must have identical shapes")
    if before_logits.ndim < 1 or before_logits.shape[-1] == 0:
        raise ValueError("Logits must have a non-empty vocabulary dimension")

    log_p = F.log_softmax(before_logits.float(), dim=-1)
    log_q = F.log_softmax(after_logits.float(), dim=-1)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    if mode == JSD_STANDARD:
        divergence = 0.5 * (
            (log_p.exp() * (log_p - log_m)).sum(dim=-1)
            + (log_q.exp() * (log_q - log_m)).sum(dim=-1)
        )
    else:
        mixture = log_m.exp()
        divergence = (
            0.5
            * (
                (mixture * (log_m - log_p)).mean(dim=-1)
                + (mixture * (log_m - log_q)).mean(dim=-1)
            )
            * LEGACY_SCALE
        )
    return divergence.clamp_min(0.0)


def streaming_js_divergence(
    adapter: DecoderModelAdapter,
    before_hidden: Tensor,
    after_hidden: Tensor,
    *,
    modes: Sequence[str] = (JSD_STANDARD,),
    vocab_chunk_size: int = 4096,
    token_chunk_size: int = 256,
    output_device: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    """Exact JSD with two vocabulary passes and bounded temporary logits.

    The returned tensors have shape ``before_hidden.shape[:-1]``.  Final norm
    and unembedding are taken from the target model, as required by LogitLens.
    """

    modes = _validate_js_modes(modes)
    if before_hidden.shape != after_hidden.shape:
        raise ValueError("before_hidden and after_hidden must have identical shapes")
    if before_hidden.ndim < 2 or before_hidden.shape[-1] != adapter.hidden_size:
        raise ValueError(
            "Hidden states must end in the adapter hidden size "
            f"{adapter.hidden_size}, got {tuple(before_hidden.shape)}"
        )
    if vocab_chunk_size <= 0 or token_chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive")

    original_shape = before_hidden.shape[:-1]
    before_flat = before_hidden.reshape(-1, adapter.hidden_size)
    after_flat = after_hidden.reshape(-1, adapter.hidden_size)
    results = {
        mode: torch.empty(before_flat.shape[0], dtype=torch.float32, device=output_device)
        for mode in modes
    }
    vocabulary_size = adapter.vocab_size

    with torch.inference_mode():
        for token_start in range(0, before_flat.shape[0], token_chunk_size):
            token_end = min(token_start + token_chunk_size, before_flat.shape[0])
            before = adapter.normalize_for_logit_lens(before_flat[token_start:token_end])
            after = adapter.normalize_for_logit_lens(after_flat[token_start:token_end])
            lm_device = adapter.output_device
            lm_dtype = adapter.lm_head.weight.dtype
            before = before.to(device=lm_device, dtype=lm_dtype)
            after = after.to(device=lm_device, dtype=lm_dtype)
            count = token_end - token_start

            # Fast path used by the target configs: a small token chunk keeps
            # two full-vocabulary logits tensors affordable, and each hidden
            # state is projected only once.  The blockwise fallback below is
            # lower-memory but necessarily repeats projection after finding
            # the two partition functions.
            if vocab_chunk_size >= vocabulary_size:
                before_logits = adapter.project_logits(
                    before,
                    0,
                    vocabulary_size,
                ).float()
                after_logits = adapter.project_logits(
                    after,
                    0,
                    vocabulary_size,
                ).float()
                log_p = F.log_softmax(before_logits, dim=-1)
                log_q = F.log_softmax(after_logits, dim=-1)
                log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
                if JSD_STANDARD in results:
                    values = 0.5 * (
                        (log_p.exp() * (log_p - log_m)).sum(dim=-1)
                        + (log_q.exp() * (log_q - log_m)).sum(dim=-1)
                    )
                    results[JSD_STANDARD][token_start:token_end] = (
                        values.clamp_min(0.0).to(output_device)
                    )
                if JSD_LEGACY_REDEEP in results:
                    mixture = log_m.exp()
                    values = (
                        0.5
                        * (
                            (mixture * (log_m - log_p)).mean(dim=-1)
                            + (mixture * (log_m - log_q)).mean(dim=-1)
                        )
                        * LEGACY_SCALE
                    )
                    results[JSD_LEGACY_REDEEP][token_start:token_end] = (
                        values.clamp_min(0.0).to(output_device)
                    )
                del before_logits, after_logits, log_p, log_q, log_m
                continue

            log_z_before = torch.full(
                (count,),
                -torch.inf,
                dtype=torch.float32,
                device=lm_device,
            )
            log_z_after = torch.full_like(log_z_before, -torch.inf)

            for vocab_start in range(0, vocabulary_size, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocabulary_size)
                before_logits = adapter.project_logits(before, vocab_start, vocab_end).float()
                after_logits = adapter.project_logits(after, vocab_start, vocab_end).float()
                log_z_before = torch.logaddexp(
                    log_z_before,
                    torch.logsumexp(before_logits, dim=-1),
                )
                log_z_after = torch.logaddexp(
                    log_z_after,
                    torch.logsumexp(after_logits, dim=-1),
                )

            accumulators = {
                mode: torch.zeros(count, dtype=torch.float32, device=lm_device)
                for mode in modes
            }
            for vocab_start in range(0, vocabulary_size, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocabulary_size)
                before_logits = adapter.project_logits(before, vocab_start, vocab_end).float()
                after_logits = adapter.project_logits(after, vocab_start, vocab_end).float()
                log_p = before_logits - log_z_before[:, None]
                log_q = after_logits - log_z_after[:, None]
                log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
                if JSD_STANDARD in accumulators:
                    accumulators[JSD_STANDARD].add_(
                        0.5
                        * (
                            (log_p.exp() * (log_p - log_m)).sum(dim=-1)
                            + (log_q.exp() * (log_q - log_m)).sum(dim=-1)
                        )
                    )
                if JSD_LEGACY_REDEEP in accumulators:
                    mixture = log_m.exp()
                    accumulators[JSD_LEGACY_REDEEP].add_(
                        0.5
                        * (
                            (mixture * (log_m - log_p)).sum(dim=-1)
                            + (mixture * (log_m - log_q)).sum(dim=-1)
                        )
                    )

            if JSD_LEGACY_REDEEP in accumulators:
                accumulators[JSD_LEGACY_REDEEP].mul_(LEGACY_SCALE / vocabulary_size)
            for mode, values in accumulators.items():
                results[mode][token_start:token_end] = (
                    values.clamp_min(0.0).to(output_device)
                )

    return {mode: values.reshape(original_shape) for mode, values in results.items()}


def compute_pks_modes(
    adapter: DecoderModelAdapter,
    before_by_layer: Mapping[int, Tensor],
    after_by_layer: Mapping[int, Tensor],
    *,
    modes: Sequence[str] = (JSD_STANDARD,),
    vocab_chunk_size: int = 4096,
    token_chunk_size: int = 256,
    output_device: str | torch.device = "cpu",
) -> dict[str, dict[int, Tensor]]:
    """Compute PKS for each captured layer without retaining layer logits."""

    modes = _validate_js_modes(modes)
    if set(before_by_layer) != set(after_by_layer):
        raise ValueError("Before/after residual mappings must have the same layer keys")
    output = {mode: {} for mode in modes}
    for layer in sorted(before_by_layer):
        layer_scores = streaming_js_divergence(
            adapter,
            before_by_layer[layer],
            after_by_layer[layer],
            modes=modes,
            vocab_chunk_size=vocab_chunk_size,
            token_chunk_size=token_chunk_size,
            output_device=output_device,
        )
        for mode in modes:
            output[mode][layer] = layer_scores[mode]
    return output


def compute_pks(
    adapter: DecoderModelAdapter,
    before_by_layer: Mapping[int, Tensor],
    after_by_layer: Mapping[int, Tensor],
    *,
    mode: str = JSD_STANDARD,
    vocab_chunk_size: int = 4096,
    token_chunk_size: int = 256,
    output_device: str | torch.device = "cpu",
) -> dict[int, Tensor]:
    """Single-mode convenience wrapper around :func:`compute_pks_modes`."""

    return compute_pks_modes(
        adapter,
        before_by_layer,
        after_by_layer,
        modes=(mode,),
        vocab_chunk_size=vocab_chunk_size,
        token_chunk_size=token_chunk_size,
        output_device=output_device,
    )[mode]


def recompute_selected_attention(
    adapter: DecoderModelAdapter,
    *,
    layer_index: int,
    query_inputs: Tensor,
    key_inputs: Tensor,
    query_positions: Sequence[int] | Tensor,
    key_positions: Sequence[int] | Tensor,
    head_indices: Sequence[int],
    key_attention_mask: Tensor | None = None,
    normalize: bool = True,
) -> Tensor:
    """Public feature-layer wrapper for the adapter's QK reconstruction."""

    return adapter.recompute_attention(
        layer_index=layer_index,
        query_inputs=query_inputs,
        key_inputs=key_inputs,
        query_positions=query_positions,
        key_positions=key_positions,
        head_indices=head_indices,
        key_attention_mask=key_attention_mask,
        normalize=normalize,
    )


def compute_ecs(
    attention_scores: Tensor,
    query_hidden: Tensor,
    context_hidden: Tensor,
    *,
    valid_key_mask: Tensor | None = None,
    top_fraction: float = 0.1,
    query_chunk_size: int = 16,
    output_device: str | torch.device = "cpu",
) -> Tensor:
    """Compute the paper's External Context Score for selected heads.

    Args:
        attention_scores: ``[batch, heads, queries, context_keys]``.  Scores may
            be raw QK logits or probabilities because only their ranking is used.
        query_hidden: Last decoder-layer residuals, ``[batch, queries, hidden]``.
        context_hidden: Final-normalized decoder hidden states,
            ``[batch, context_keys, hidden]``.
        valid_key_mask: Optional causal/padding mask ``[batch, queries,
            context_keys]``.  This is required when the key universe contains
            future response positions (the whole-prefix compatibility mode).

    Returns:
        Cosine similarities with shape ``[batch, heads, queries]``.
    """

    if attention_scores.ndim != 4:
        raise ValueError("attention_scores must have shape [batch, heads, queries, keys]")
    if query_hidden.ndim != 3 or context_hidden.ndim != 3:
        raise ValueError("query_hidden and context_hidden must be three-dimensional")
    batch, heads, queries, keys = attention_scores.shape
    if tuple(query_hidden.shape[:2]) != (batch, queries):
        raise ValueError("query_hidden does not align with attention query dimensions")
    if tuple(context_hidden.shape[:2]) != (batch, keys):
        raise ValueError("context_hidden does not align with attention key dimensions")
    if query_hidden.shape[-1] != context_hidden.shape[-1]:
        raise ValueError("query/context hidden dimensions differ")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if keys == 0:
        raise ValueError("At least one context key is required")

    compute_device = attention_scores.device
    query_hidden = query_hidden.to(compute_device)
    context_hidden = context_hidden.to(compute_device)
    if valid_key_mask is None:
        valid_key_mask = torch.ones(
            (batch, queries, keys),
            dtype=torch.bool,
            device=compute_device,
        )
    else:
        if tuple(valid_key_mask.shape) != (batch, queries, keys):
            raise ValueError(
                "valid_key_mask must have shape "
                f"{(batch, queries, keys)}, got {tuple(valid_key_mask.shape)}"
            )
        valid_key_mask = valid_key_mask.to(device=compute_device, dtype=torch.bool)
    valid_counts = valid_key_mask.sum(dim=-1)
    result = torch.empty(
        (batch, heads, queries),
        dtype=torch.float32,
        device=output_device,
    )
    batch_index = torch.arange(batch, device=compute_device)[:, None, None]
    with torch.inference_mode():
        for head in range(heads):
            for query_start in range(0, queries, query_chunk_size):
                query_end = min(query_start + query_chunk_size, queries)
                chunk_mask = valid_key_mask[:, query_start:query_end, :]
                chunk_counts = valid_counts[:, query_start:query_end]
                requested_counts = torch.floor(chunk_counts * top_fraction).to(
                    torch.long
                )
                requested_counts = torch.where(
                    chunk_counts > 0,
                    requested_counts.clamp_min(1),
                    torch.zeros_like(requested_counts),
                )
                maximum_count = max(1, int(requested_counts.max().item()))
                chunk_scores = attention_scores[
                    :,
                    head,
                    query_start:query_end,
                    :,
                ].masked_fill(~chunk_mask, -torch.inf)
                top_indices = chunk_scores.topk(maximum_count, dim=-1).indices
                selected_context = context_hidden[batch_index, top_indices]
                rank = torch.arange(maximum_count, device=compute_device)[None, None, :]
                selected_mask = rank < requested_counts[:, :, None]
                denominator = requested_counts.clamp_min(1).float()[:, :, None]
                pooled_context = (
                    selected_context.float() * selected_mask[:, :, :, None]
                ).sum(dim=2) / denominator
                current_query = query_hidden[:, query_start:query_end, :].float()
                similarity = F.cosine_similarity(pooled_context, current_query, dim=-1)
                similarity = similarity.masked_fill(requested_counts == 0, torch.nan)
                result[:, head, query_start:query_end] = similarity.to(output_device)
    return result


@dataclass
class TokenFeatureBatch:
    """Feature columns for a batch sharing the same absolute token positions."""

    predictor_positions: tuple[int, ...]
    columns: dict[str, Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_tokens = len(self.predictor_positions)
        batch_size: int | None = None
        for name, values in self.columns.items():
            if values.ndim != 2 or values.shape[1] != expected_tokens:
                raise ValueError(
                    f"Feature {name} has shape {tuple(values.shape)}; "
                    f"expected [batch, {expected_tokens}]"
                )
            if batch_size is None:
                batch_size = values.shape[0]
            elif values.shape[0] != batch_size:
                raise ValueError("All feature columns must have the same batch size")


def _validated_positions(name: str, positions: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(position) for position in positions)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if any(position < 0 for position in result):
        raise ValueError(f"{name} cannot contain negative positions")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicate positions")
    return result


def _position_indices(
    positions: Sequence[int],
    universe: Sequence[int],
    *,
    device: torch.device,
) -> Tensor:
    lookup = {position: index for index, position in enumerate(universe)}
    try:
        indices = [lookup[position] for position in positions]
    except KeyError as error:
        raise ValueError(
            f"Position {error.args[0]} is absent from the captured universe"
        ) from error
    return torch.tensor(indices, dtype=torch.long, device=device)


def _valid_key_mask(
    query_positions: Sequence[int] | Tensor,
    key_positions: Sequence[int] | Tensor,
    *,
    batch_size: int,
    key_attention_mask: Tensor | None,
    sliding_window: int | None,
    device: torch.device,
) -> Tensor:
    query = torch.as_tensor(query_positions, dtype=torch.long, device=device)
    key = torch.as_tensor(key_positions, dtype=torch.long, device=device)
    if query.ndim == 1:
        query = query.unsqueeze(0).expand(batch_size, -1)
    if key.ndim == 1:
        key = key.unsqueeze(0).expand(batch_size, -1)
    if query.ndim != 2 or query.shape[0] != batch_size:
        raise ValueError("query_positions do not align with the feature batch")
    if key.ndim != 2 or key.shape[0] != batch_size:
        raise ValueError("key_positions do not align with the feature batch")
    valid = key[:, None, :] <= query[:, :, None]
    if sliding_window is not None:
        valid &= key[:, None, :] > query[:, :, None] - int(sliding_window)
    if key_attention_mask is not None:
        valid &= key_attention_mask.to(device=device, dtype=torch.bool)[:, None, :]
    return valid


def extract_token_features(
    adapter: DecoderModelAdapter,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    predictor_positions: Sequence[int],
    context_positions: Sequence[int],
    copying_heads: Sequence[tuple[int, int]],
    *,
    position_ids: Tensor | None = None,
    pks_layers: Sequence[int] | None = None,
    pks_modes: Sequence[str] = (JSD_STANDARD,),
    top_fraction: float = 0.1,
    vocab_chunk_size: int = 4096,
    token_chunk_size: int = 256,
    ecs_query_chunk_size: int = 16,
    whole_prefix_positions: Sequence[int] | None = None,
    offload_to_cpu: bool = True,
) -> TokenFeatureBatch:
    """Extract all requested token features with one target-model forward pass.

    The caller supplies absolute sequence indices.  In the reproduction these
    are ``TokenizedExample.predictor_positions`` and
    ``TokenizedExample.context_token_positions``.  ``whole_prefix_positions``
    enables the released code's fixed-prompt-prefix ECS compatibility mode. It
    should contain only tokens before the first response token.
    """

    predictors = _validated_positions("predictor_positions", predictor_positions)
    context = _validated_positions("context_positions", context_positions)
    whole_prefix = (
        None
        if whole_prefix_positions is None
        else _validated_positions("whole_prefix_positions", whole_prefix_positions)
    )
    if whole_prefix is not None and max(whole_prefix) > min(predictors):
        raise ValueError(
            "whole_prefix_positions must end at or before the first response "
            "predictor; prior response tokens are not part of the public-code prefix"
        )
    if max((*predictors, *context, *(whole_prefix or ()))) >= input_ids.shape[1]:
        raise IndexError("A requested position is outside input_ids")

    heads = tuple((int(layer), int(head)) for layer, head in copying_heads)
    if len(set(heads)) != len(heads):
        raise ValueError("copying_heads cannot contain duplicates")
    for layer, head in heads:
        if not 0 <= layer < adapter.num_layers:
            raise IndexError(f"Invalid copying-head layer {layer}")
        if not 0 <= head < adapter.num_attention_heads:
            raise IndexError(f"Invalid copying-head index {head}")
    heads_by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        heads_by_layer.setdefault(layer, []).append(head)

    modes = tuple(pks_modes)
    if modes:
        modes = _validate_js_modes(modes)
        layers = (
            tuple(range(adapter.num_layers))
            if pks_layers is None
            else tuple(int(layer) for layer in pks_layers)
        )
        if len(set(layers)) != len(layers):
            raise ValueError("pks_layers cannot contain duplicates")
        if any(not 0 <= layer < adapter.num_layers for layer in layers):
            raise IndexError("pks_layers contains an invalid layer")
    else:
        layers = ()

    key_universe = tuple(dict.fromkeys((*context, *(whole_prefix or ()))))
    hidden_universe = tuple(dict.fromkeys((*predictors, *key_universe)))
    outputs, captured = adapter.run_with_capture(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        ffn_positions=predictors if layers else (),
        ffn_layers=layers,
        attention_layers=tuple(sorted(heads_by_layer)),
        attention_query_positions=predictors if heads else (),
        attention_key_positions=key_universe if heads else (),
        last_hidden_positions=hidden_universe if heads else (),
        offload_to_cpu=offload_to_cpu,
    )
    del outputs
    columns: dict[str, Tensor] = {}

    if layers:
        pks = compute_pks_modes(
            adapter,
            captured.pre_ffn,
            captured.post_ffn,
            modes=modes,
            vocab_chunk_size=vocab_chunk_size,
            token_chunk_size=token_chunk_size,
            output_device="cpu",
        )
        for mode in modes:
            for layer in layers:
                columns[pks_feature_name(mode, layer)] = pks[mode][layer].cpu()

    if heads:
        if captured.last_hidden is None:
            raise RuntimeError("Last-layer hidden states were not captured")
        # HF ``hidden_states[-1]`` / ``last_hidden_state`` include final RMSNorm.
        # Hooks observe the raw final-layer residual, so match HF before ECS.
        with torch.inference_mode():
            final_hidden = adapter.normalize_for_logit_lens(captured.last_hidden)
        query_hidden_index = _position_indices(
            predictors,
            hidden_universe,
            device=final_hidden.device,
        )
        context_hidden_index = _position_indices(
            context,
            hidden_universe,
            device=final_hidden.device,
        )
        query_hidden = final_hidden.index_select(1, query_hidden_index)
        context_hidden = final_hidden.index_select(1, context_hidden_index)
        context_key_index: Tensor | None = None
        whole_key_index: Tensor | None = None

        for layer, layer_heads in sorted(heads_by_layer.items()):
            query_inputs = captured.attention_queries[layer]
            key_inputs = captured.attention_keys[layer]
            key_mask = None
            if attention_mask is not None:
                key_position_index = torch.tensor(
                    key_universe,
                    dtype=torch.long,
                    device=attention_mask.device,
                )
                key_mask = attention_mask.index_select(1, key_position_index)
            query_absolute = (
                predictors
                if position_ids is None
                else position_ids.index_select(
                    1,
                    torch.tensor(
                        predictors,
                        dtype=torch.long,
                        device=position_ids.device,
                    ),
                )
            )
            key_absolute = (
                key_universe
                if position_ids is None
                else position_ids.index_select(
                    1,
                    torch.tensor(
                        key_universe,
                        dtype=torch.long,
                        device=position_ids.device,
                    ),
                )
            )
            raw_attention = recompute_selected_attention(
                adapter,
                layer_index=layer,
                query_inputs=query_inputs,
                key_inputs=key_inputs,
                query_positions=query_absolute,
                key_positions=key_absolute,
                head_indices=layer_heads,
                key_attention_mask=key_mask,
                normalize=False,
            )
            valid_universe = _valid_key_mask(
                query_absolute,
                key_absolute,
                batch_size=input_ids.shape[0],
                key_attention_mask=key_mask,
                sliding_window=adapter.attention_geometry(layer).sliding_window,
                device=raw_attention.device,
            )
            if context_key_index is None:
                context_key_index = _position_indices(
                    context,
                    key_universe,
                    device=raw_attention.device,
                )
            context_attention = raw_attention.index_select(-1, context_key_index)
            context_ecs = compute_ecs(
                context_attention,
                query_hidden,
                context_hidden,
                valid_key_mask=valid_universe.index_select(-1, context_key_index),
                top_fraction=top_fraction,
                query_chunk_size=ecs_query_chunk_size,
                output_device="cpu",
            )
            for local_head, head in enumerate(layer_heads):
                columns[ecs_feature_name(layer, head)] = context_ecs[:, local_head, :]

            if whole_prefix is not None:
                if whole_key_index is None:
                    whole_key_index = _position_indices(
                        whole_prefix,
                        key_universe,
                        device=raw_attention.device,
                    )
                whole_hidden_index = _position_indices(
                    whole_prefix,
                    hidden_universe,
                    device=final_hidden.device,
                )
                whole_hidden = final_hidden.index_select(1, whole_hidden_index)
                whole_attention = raw_attention.index_select(-1, whole_key_index)
                whole_ecs = compute_ecs(
                    whole_attention,
                    query_hidden,
                    whole_hidden,
                    valid_key_mask=valid_universe.index_select(-1, whole_key_index),
                    top_fraction=top_fraction,
                    query_chunk_size=ecs_query_chunk_size,
                    output_device="cpu",
                )
                for local_head, head in enumerate(layer_heads):
                    columns[
                        ecs_feature_name(layer, head, whole_prefix=True)
                    ] = whole_ecs[:, local_head, :]

    metadata: dict[str, Any] = {
        "model_type": adapter.model_type,
        "predictor_positions": list(predictors),
        "context_positions": list(context),
        "whole_prefix_positions": (
            None if whole_prefix is None else list(whole_prefix)
        ),
        "copying_heads": [list(head) for head in heads],
        "pks_layers": list(layers),
        "pks_modes": list(modes),
        "top_fraction": float(top_fraction),
        "top_count_context": max(1, math.floor(len(context) * top_fraction)),
        "top_count_rounding": "floor with minimum one valid key",
        "jsd_standard": "0.5*KL(P||M)+0.5*KL(Q||M)",
        "jsd_legacy_redeep": (
            "mean_vocab(0.5*KL(M||P)+0.5*KL(M||Q))*1e6"
        ),
        "attention_normalization": (
            "raw selected-key QK logits; ECS uses rank only; causal/padding masked"
        ),
    }
    return TokenFeatureBatch(
        predictor_positions=predictors,
        columns=columns,
        metadata=metadata,
    )
