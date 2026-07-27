"""Memory-bounded extraction of the released ReDeEP ECS and PKS."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch.nn import functional as functional


def released_reverse_jsd(
    post_ffn_logits: torch.Tensor,
    pre_ffn_logits: torch.Tensor,
) -> torch.Tensor:
    """Match the non-standard divergence in the released ReDeEP code.

    The upstream implementation computes KL(M || P) and KL(M || Q), averages
    over vocabulary entries, and multiplies by ``10e5``.  This is intentionally
    not silently replaced with standard Jensen-Shannon divergence.
    """

    post_distribution = functional.softmax(post_ffn_logits, dim=-1)
    pre_distribution = functional.softmax(pre_ffn_logits, dim=-1)
    midpoint = 0.5 * (post_distribution + pre_distribution)
    post_term = functional.kl_div(
        functional.log_softmax(post_ffn_logits, dim=-1),
        midpoint,
        reduction="none",
    ).mean(-1)
    pre_term = functional.kl_div(
        functional.log_softmax(pre_ffn_logits, dim=-1),
        midpoint,
        reduction="none",
    ).mean(-1)
    return 0.5 * (post_term + pre_term) * 1_000_000.0


def top_context_indices(
    attention_logits: torch.Tensor,
    *,
    top_fraction: float = 0.1,
) -> torch.Tensor:
    """Select the same context positions as sorting attention probabilities."""

    if attention_logits.ndim < 2:
        raise ValueError("attention logits must have a context dimension")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must lie in (0, 1]")
    context_length = attention_logits.shape[-1]
    top_count = int(context_length * top_fraction)
    if top_count < 1:
        raise ValueError(
            "the context is too short for the released floor(top_fraction*N)"
        )
    return torch.argsort(
        attention_logits,
        dim=-1,
        descending=True,
    )[..., :top_count]


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first = values[..., : values.shape[-1] // 2]
    second = values[..., values.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    values: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return values * cosine + _rotate_half(values) * sine


def project_query_key(
    attention: Any,
    *,
    query_inputs: torch.Tensor,
    key_inputs: torch.Tensor,
    num_query_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project normalized attention inputs using the model's native Q/K path.

    Qwen3 applies per-head Q/K normalization after projection and before the
    head transpose.  Llama has no such modules, so the same helper covers both
    architectures without changing their respective computation graphs.
    """

    batch_size, query_length, _hidden_size = query_inputs.shape
    key_length = key_inputs.shape[1]
    queries = attention.q_proj(query_inputs).view(
        batch_size,
        query_length,
        num_query_heads,
        head_dim,
    )
    keys = attention.k_proj(key_inputs).view(
        batch_size,
        key_length,
        num_key_value_heads,
        head_dim,
    )
    query_norm = getattr(attention, "q_norm", None)
    key_norm = getattr(attention, "k_norm", None)
    if query_norm is not None:
        queries = query_norm(queries)
    if key_norm is not None:
        keys = key_norm(keys)
    return queries.transpose(1, 2), keys.transpose(1, 2)


@dataclass
class _ExtractionState:
    top_indices: dict[int, torch.Tensor] = field(default_factory=dict)
    pre_ffn: dict[int, torch.Tensor] = field(default_factory=dict)
    post_ffn: dict[int, torch.Tensor] = field(default_factory=dict)
    final_hidden: torch.Tensor | None = None


class ReDeEPFeatureExtractor:
    """Extract exact released-code features without retaining full attentions."""

    def __init__(
        self,
        model: Any,
        *,
        candidate_heads: Sequence[Sequence[int]],
        top_fraction: float = 0.1,
        logit_chunk_size: int = 4,
        cosine_chunk_size: int = 8,
    ) -> None:
        model_type = str(getattr(model.config, "model_type", "")).lower()
        if model_type not in {"llama", "qwen3"}:
            raise ValueError(
                "this extractor supports only Llama and Qwen3"
            )
        if logit_chunk_size <= 0 or cosine_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")
        self.model = model
        self.layers = model.model.layers
        self.candidate_heads = [
            (int(pair[0]), int(pair[1])) for pair in candidate_heads
        ]
        self.top_fraction = float(top_fraction)
        self.logit_chunk_size = int(logit_chunk_size)
        self.cosine_chunk_size = int(cosine_chunk_size)
        self._heads_by_layer: dict[int, list[tuple[int, int]]] = defaultdict(
            list
        )
        num_layers = len(self.layers)
        num_heads = int(model.config.num_attention_heads)
        for candidate_index, (layer_index, head_index) in enumerate(
            self.candidate_heads
        ):
            if not 0 <= layer_index < num_layers:
                raise ValueError(f"invalid candidate layer {layer_index}")
            if not 0 <= head_index < num_heads:
                raise ValueError(f"invalid candidate head {head_index}")
            self._heads_by_layer[layer_index].append(
                (candidate_index, head_index)
            )

    def _attention_hook(
        self,
        *,
        layer_index: int,
        prefix_positions: torch.Tensor,
        score_positions: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        state: _ExtractionState,
    ):
        candidate_pairs = self._heads_by_layer[layer_index]

        def hook(module: Any, arguments: tuple[Any, ...]) -> None:
            hidden_states = arguments[0]
            normalized_prefix = module.input_layernorm(
                hidden_states.index_select(1, prefix_positions)
            )
            normalized_queries = module.input_layernorm(
                hidden_states.index_select(1, score_positions)
            )
            attention = module.self_attn
            batch_size = hidden_states.shape[0]
            query_length = score_positions.numel()
            prefix_length = prefix_positions.numel()
            head_dim = int(attention.head_dim)
            num_heads = int(self.model.config.num_attention_heads)
            num_key_value_heads = int(
                self.model.config.num_key_value_heads
            )
            queries, keys = project_query_key(
                attention,
                query_inputs=normalized_queries,
                key_inputs=normalized_prefix,
                num_query_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
            )
            queries = _apply_rope(
                queries,
                cosine.index_select(1, score_positions),
                sine.index_select(1, score_positions),
            )
            keys = _apply_rope(
                keys,
                cosine.index_select(1, prefix_positions),
                sine.index_select(1, prefix_positions),
            )

            candidate_indices = [
                candidate_index
                for candidate_index, _head_index in candidate_pairs
            ]
            head_indices = torch.tensor(
                [
                    head_index
                    for _candidate_index, head_index in candidate_pairs
                ],
                device=hidden_states.device,
                dtype=torch.long,
            )
            key_value_indices = torch.div(
                head_indices,
                int(attention.num_key_value_groups),
                rounding_mode="floor",
            )
            selected_queries = queries.index_select(1, head_indices)
            selected_keys = keys.index_select(1, key_value_indices)
            attention_logits = torch.matmul(
                selected_queries,
                selected_keys.transpose(-2, -1),
            ) * float(attention.scaling)
            attention_probabilities = torch.softmax(
                attention_logits,
                dim=-1,
                dtype=torch.float32,
            ).to(dtype=attention_logits.dtype)
            selected_positions = top_context_indices(
                attention_probabilities,
                top_fraction=self.top_fraction,
            )
            for local_index, candidate_index in enumerate(candidate_indices):
                state.top_indices[candidate_index] = (
                    selected_positions[0, local_index]
                    .detach()
                    .to(device="cpu", dtype=torch.int32)
                )

        return hook

    @staticmethod
    def _pre_ffn_hook(
        layer_index: int,
        score_positions: torch.Tensor,
        state: _ExtractionState,
    ):
        def hook(_module: Any, arguments: tuple[Any, ...]) -> None:
            state.pre_ffn[layer_index] = (
                arguments[0]
                .index_select(1, score_positions)
                .detach()
                .clone()
            )

        return hook

    @staticmethod
    def _post_ffn_hook(
        layer_index: int,
        score_positions: torch.Tensor,
        state: _ExtractionState,
    ):
        def hook(
            _module: Any,
            _arguments: tuple[Any, ...],
            output: Any,
        ) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            state.post_ffn[layer_index] = (
                hidden_states.index_select(1, score_positions)
                .detach()
                .clone()
            )

        return hook

    @staticmethod
    def _final_hidden_hook(state: _ExtractionState):
        def hook(
            _module: Any,
            _arguments: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            state.final_hidden = output.detach().clone()

        return hook

    def _compute_external_scores(
        self,
        *,
        state: _ExtractionState,
        score_positions: torch.Tensor,
    ) -> torch.Tensor:
        if state.final_hidden is None:
            raise RuntimeError("final hidden states were not captured")
        final_hidden = state.final_hidden[0]
        token_count = score_positions.numel()
        external = torch.empty(
            (token_count, len(self.candidate_heads)),
            dtype=torch.float32,
            device="cpu",
        )
        for candidate_index in range(len(self.candidate_heads)):
            if candidate_index not in state.top_indices:
                raise RuntimeError(
                    f"candidate head {candidate_index} was not captured"
                )
            indices = state.top_indices[candidate_index]
            for start in range(0, token_count, self.cosine_chunk_size):
                end = min(start + self.cosine_chunk_size, token_count)
                selected_indices = indices[start:end].to(
                    device=final_hidden.device,
                    dtype=torch.long,
                )
                selected_hidden = final_hidden[selected_indices]
                attended_hidden = selected_hidden.mean(dim=1)
                current_hidden = final_hidden.index_select(
                    0, score_positions[start:end]
                )
                similarity = functional.cosine_similarity(
                    attended_hidden,
                    current_hidden,
                    dim=-1,
                )
                external[start:end, candidate_index] = (
                    similarity.detach().float().cpu()
                )
        return external

    def _compute_parametric_scores(
        self,
        *,
        state: _ExtractionState,
        token_count: int,
    ) -> torch.Tensor:
        parametric = torch.empty(
            (token_count, len(self.layers)),
            dtype=torch.float32,
            device="cpu",
        )
        for layer_index in range(len(self.layers)):
            try:
                pre_ffn = state.pre_ffn[layer_index]
                post_ffn = state.post_ffn[layer_index]
            except KeyError as error:
                raise RuntimeError(
                    f"FFN states for layer {layer_index} were not captured"
                ) from error
            for start in range(0, token_count, self.logit_chunk_size):
                end = min(start + self.logit_chunk_size, token_count)
                pre_logits = self.model.lm_head(
                    self.model.model.norm(pre_ffn[:, start:end, :])
                )
                post_logits = self.model.lm_head(
                    self.model.model.norm(post_ffn[:, start:end, :])
                )
                scores = released_reverse_jsd(post_logits, pre_logits)
                parametric[start:end, layer_index] = (
                    scores[0].detach().float().cpu()
                )
                del pre_logits, post_logits, scores
        return parametric

    @torch.inference_mode()
    def extract(
        self,
        *,
        input_ids: Sequence[int],
        prefix_length: int,
        score_positions: Sequence[int],
    ) -> dict[str, list[list[float]]]:
        """Extract one ECS/PKS row per fixed-response token."""

        if not input_ids:
            raise ValueError("input_ids must not be empty")
        if not 1 <= prefix_length <= len(input_ids):
            raise ValueError("prefix_length lies outside input_ids")
        normalized_score_positions = [int(value) for value in score_positions]
        if not normalized_score_positions:
            raise ValueError("score_positions must not be empty")
        if any(
            value < 0 or value >= len(input_ids)
            for value in normalized_score_positions
        ):
            raise ValueError("a score position lies outside input_ids")

        device = next(self.model.parameters()).device
        input_tensor = torch.tensor(
            [list(input_ids)],
            device=device,
            dtype=torch.long,
        )
        prefix_positions = torch.arange(
            prefix_length,
            device=device,
            dtype=torch.long,
        )
        score_position_tensor = torch.tensor(
            normalized_score_positions,
            device=device,
            dtype=torch.long,
        )
        position_ids = torch.arange(
            len(input_ids),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        model_dtype = next(self.model.parameters()).dtype
        rotary_reference = torch.empty(
            (1, len(input_ids), 1),
            device=device,
            dtype=model_dtype,
        )
        cosine, sine = self.model.model.rotary_emb(
            rotary_reference, position_ids
        )
        state = _ExtractionState()
        handles = []
        try:
            for layer_index, layer in enumerate(self.layers):
                if layer_index in self._heads_by_layer:
                    handles.append(
                        layer.register_forward_pre_hook(
                            self._attention_hook(
                                layer_index=layer_index,
                                prefix_positions=prefix_positions,
                                score_positions=score_position_tensor,
                                cosine=cosine,
                                sine=sine,
                                state=state,
                            )
                        )
                    )
                handles.append(
                    layer.post_attention_layernorm.register_forward_pre_hook(
                        self._pre_ffn_hook(
                            layer_index,
                            score_position_tensor,
                            state,
                        )
                    )
                )
                handles.append(
                    layer.register_forward_hook(
                        self._post_ffn_hook(
                            layer_index,
                            score_position_tensor,
                            state,
                        )
                    )
                )
            handles.append(
                self.model.model.norm.register_forward_hook(
                    self._final_hidden_hook(state)
                )
            )
            self.model(
                input_ids=input_tensor,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                logits_to_keep=1,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()

        external = self._compute_external_scores(
            state=state,
            score_positions=score_position_tensor,
        )
        parametric = self._compute_parametric_scores(
            state=state,
            token_count=len(normalized_score_positions),
        )
        return {
            "external": external.tolist(),
            "parametric": parametric.tolist(),
        }
