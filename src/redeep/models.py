"""Model adapters and state capture for Llama 3.1 and Qwen3 causal LMs.

The implementation intentionally relies on public module attributes shared by
``transformers==4.52.2`` instead of vendoring or patching Transformers.  No CUDA
operation is performed at import time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

SUPPORTED_MODEL_TYPES = frozenset({"llama", "qwen3"})


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _as_index_tuple(
    positions: Sequence[int] | Tensor | None,
    *,
    name: str,
) -> tuple[int, ...] | None:
    if positions is None:
        return None
    if isinstance(positions, Tensor):
        if positions.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {tuple(positions.shape)}")
        positions = positions.detach().cpu().tolist()
    result = tuple(int(position) for position in positions)
    if any(position < 0 for position in result):
        raise ValueError(f"{name} cannot contain negative positions")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicate positions")
    return result


def _select_sequence_positions(hidden: Tensor, positions: tuple[int, ...] | None) -> Tensor:
    if positions is None:
        return hidden
    if not positions:
        return hidden[:, :0, :]
    if max(positions) >= hidden.shape[1]:
        raise IndexError(
            f"Position {max(positions)} is outside a sequence of length {hidden.shape[1]}"
        )
    index = torch.tensor(positions, dtype=torch.long, device=hidden.device)
    return hidden.index_select(1, index)


def _save_tensor(hidden: Tensor, *, offload_to_cpu: bool) -> Tensor:
    hidden = hidden.detach()
    if offload_to_cpu:
        return hidden.to(device="cpu", copy=True)
    return hidden.clone()


@dataclass
class CapturedModelStates:
    """Residual and normalized attention inputs captured during one forward pass.

    All layer and head indices in this package are zero-based.  ``pre_ffn`` is
    the input to ``post_attention_layernorm`` (the residual immediately before
    the FFN), while ``post_ffn`` is the decoder-layer output after adding the
    FFN contribution.
    """

    pre_ffn: dict[int, Tensor] = field(default_factory=dict)
    post_ffn: dict[int, Tensor] = field(default_factory=dict)
    attention_queries: dict[int, Tensor] = field(default_factory=dict)
    attention_keys: dict[int, Tensor] = field(default_factory=dict)
    last_hidden: Tensor | None = None
    ffn_positions: tuple[int, ...] | None = None
    attention_query_positions: tuple[int, ...] | None = None
    attention_key_positions: tuple[int, ...] | None = None
    last_hidden_positions: tuple[int, ...] | None = None


class ModelStateCapture(AbstractContextManager["ModelStateCapture"]):
    """Scoped forward hooks used by :class:`DecoderModelAdapter`.

    Passing ``None`` for a position set captures the full sequence.  Passing an
    empty tuple disables the corresponding capture.
    """

    def __init__(
        self,
        adapter: DecoderModelAdapter,
        *,
        ffn_positions: Sequence[int] | Tensor | None = (),
        ffn_layers: Sequence[int] | None = None,
        attention_layers: Sequence[int] = (),
        attention_query_positions: Sequence[int] | Tensor | None = (),
        attention_key_positions: Sequence[int] | Tensor | None = (),
        last_hidden_positions: Sequence[int] | Tensor | None = (),
        offload_to_cpu: bool = False,
    ) -> None:
        self.adapter = adapter
        self.ffn_positions = _as_index_tuple(ffn_positions, name="ffn_positions")
        self.ffn_layers = (
            tuple(range(adapter.num_layers))
            if ffn_layers is None
            else tuple(int(layer) for layer in ffn_layers)
        )
        self.attention_layers = tuple(int(layer) for layer in attention_layers)
        self.attention_query_positions = _as_index_tuple(
            attention_query_positions,
            name="attention_query_positions",
        )
        self.attention_key_positions = _as_index_tuple(
            attention_key_positions,
            name="attention_key_positions",
        )
        self.last_hidden_positions = _as_index_tuple(
            last_hidden_positions,
            name="last_hidden_positions",
        )
        self.offload_to_cpu = bool(offload_to_cpu)
        self.state = CapturedModelStates(
            ffn_positions=self.ffn_positions,
            attention_query_positions=self.attention_query_positions,
            attention_key_positions=self.attention_key_positions,
            last_hidden_positions=self.last_hidden_positions,
        )
        self._handles: list[Any] = []
        self._validate_layers()

    def _validate_layers(self) -> None:
        for name, indices in (
            ("ffn_layers", self.ffn_layers),
            ("attention_layers", self.attention_layers),
        ):
            if len(set(indices)) != len(indices):
                raise ValueError(f"{name} cannot contain duplicate indices")
            invalid = [index for index in indices if not 0 <= index < self.adapter.num_layers]
            if invalid:
                raise IndexError(f"{name} contains invalid layer indices: {invalid}")

    def __enter__(self) -> ModelStateCapture:
        if self._handles:
            raise RuntimeError("A ModelStateCapture object cannot be entered twice")

        if self.ffn_positions != ():
            for layer_index in self.ffn_layers:
                layer = self.adapter.layers[layer_index]

                def pre_ffn_hook(
                    _module: nn.Module,
                    args: tuple[Any, ...],
                    *,
                    index: int = layer_index,
                ) -> None:
                    if not args or not isinstance(args[0], Tensor):
                        raise RuntimeError("post_attention_layernorm did not receive a tensor")
                    selected = _select_sequence_positions(args[0], self.ffn_positions)
                    self.state.pre_ffn[index] = _save_tensor(
                        selected,
                        offload_to_cpu=self.offload_to_cpu,
                    )

                def post_layer_hook(
                    _module: nn.Module,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    index: int = layer_index,
                ) -> None:
                    hidden = output[0] if isinstance(output, tuple) else output
                    if not isinstance(hidden, Tensor):
                        raise RuntimeError("decoder layer did not return a hidden-state tensor")
                    selected = _select_sequence_positions(hidden, self.ffn_positions)
                    self.state.post_ffn[index] = _save_tensor(
                        selected,
                        offload_to_cpu=self.offload_to_cpu,
                    )
                    if (
                        index == self.adapter.num_layers - 1
                        and self.last_hidden_positions != ()
                    ):
                        final_selected = _select_sequence_positions(
                            hidden,
                            self.last_hidden_positions,
                        )
                        self.state.last_hidden = _save_tensor(
                            final_selected,
                            offload_to_cpu=self.offload_to_cpu,
                        )

                self._handles.append(
                    layer.post_attention_layernorm.register_forward_pre_hook(pre_ffn_hook)
                )
                self._handles.append(layer.register_forward_hook(post_layer_hook))
        if (
            self.last_hidden_positions != ()
            and (
                self.ffn_positions == ()
                or self.adapter.num_layers - 1 not in self.ffn_layers
            )
        ):
            final_layer = self.adapter.layers[-1]

            def final_layer_hook(
                _module: nn.Module,
                _args: tuple[Any, ...],
                output: Any,
            ) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, Tensor):
                    raise RuntimeError("decoder layer did not return a hidden-state tensor")
                selected = _select_sequence_positions(hidden, self.last_hidden_positions)
                self.state.last_hidden = _save_tensor(
                    selected,
                    offload_to_cpu=self.offload_to_cpu,
                )

            self._handles.append(final_layer.register_forward_hook(final_layer_hook))

        if self.attention_layers:
            if self.attention_query_positions == () or self.attention_key_positions == ():
                raise ValueError(
                    "attention query/key positions cannot be empty when attention layers are set"
                )
            for layer_index in self.attention_layers:
                attention = self.adapter.layers[layer_index].self_attn

                def attention_pre_hook(
                    _module: nn.Module,
                    args: tuple[Any, ...],
                    kwargs: Mapping[str, Any],
                    *,
                    index: int = layer_index,
                ) -> None:
                    hidden = kwargs.get("hidden_states")
                    if hidden is None and args:
                        hidden = args[0]
                    if not isinstance(hidden, Tensor):
                        raise RuntimeError("self_attn did not receive a hidden-state tensor")
                    query = _select_sequence_positions(
                        hidden,
                        self.attention_query_positions,
                    )
                    key = _select_sequence_positions(
                        hidden,
                        self.attention_key_positions,
                    )
                    self.state.attention_queries[index] = _save_tensor(
                        query,
                        offload_to_cpu=self.offload_to_cpu,
                    )
                    self.state.attention_keys[index] = _save_tensor(
                        key,
                        offload_to_cpu=self.offload_to_cpu,
                    )

                self._handles.append(
                    attention.register_forward_pre_hook(
                        attention_pre_hook,
                        with_kwargs=True,
                    )
                )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        return None


@dataclass(frozen=True)
class AttentionGeometry:
    num_query_heads: int
    num_key_value_heads: int
    num_key_value_groups: int
    head_dim: int
    scaling: float
    sliding_window: int | None


@dataclass(frozen=True)
class AttentionParityReport:
    """Numerical comparison between reconstructed and HF eager attention."""

    allclose: bool
    max_absolute_error: float
    max_relative_error: float
    recomputed: Tensor
    eager: Tensor


class DecoderModelAdapter:
    """Thin adapter over Transformers Llama/Qwen3 decoder-only language models."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.config = getattr(model, "config", None)
        self.model_type = str(getattr(self.config, "model_type", "")).lower()
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type={self.model_type!r}; "
                f"expected one of {sorted(SUPPORTED_MODEL_TYPES)}"
            )

        if hasattr(model, "get_decoder"):
            self.decoder = model.get_decoder()
        else:
            self.decoder = getattr(model, "model", None)
        if self.decoder is None:
            raise TypeError("The causal LM does not expose a decoder")

        self.layers = getattr(self.decoder, "layers", None)
        self.final_norm = getattr(self.decoder, "norm", None)
        self.embed_tokens = getattr(self.decoder, "embed_tokens", None)
        self.lm_head = (
            model.get_output_embeddings()
            if hasattr(model, "get_output_embeddings")
            else getattr(model, "lm_head", None)
        )
        if self.layers is None or self.final_norm is None:
            raise TypeError("Decoder must expose layers and a final norm")
        if self.embed_tokens is None or self.lm_head is None:
            raise TypeError("Model must expose input embeddings and an LM head")
        self._validate_architecture()

    @classmethod
    def from_model(cls, model: nn.Module) -> DecoderModelAdapter:
        return cls(model)

    def _validate_architecture(self) -> None:
        required_layer_attributes = (
            "self_attn",
            "post_attention_layernorm",
        )
        required_attention_attributes = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        )
        for layer_index, layer in enumerate(self.layers):
            for attribute in required_layer_attributes:
                if not hasattr(layer, attribute):
                    raise TypeError(f"Layer {layer_index} has no {attribute}")
            for attribute in required_attention_attributes:
                if not hasattr(layer.self_attn, attribute):
                    raise TypeError(f"Attention layer {layer_index} has no {attribute}")

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_attention_heads(self) -> int:
        return int(self.config.num_attention_heads)

    @property
    def num_key_value_heads(self) -> int:
        return int(self.config.num_key_value_heads)

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def head_dim(self) -> int:
        return int(getattr(self.config, "head_dim", self.hidden_size // self.num_attention_heads))

    @property
    def vocab_size(self) -> int:
        return int(self.lm_head.weight.shape[0])

    @property
    def output_device(self) -> torch.device:
        return _module_device(self.lm_head)

    def attention_geometry(self, layer_index: int) -> AttentionGeometry:
        attention = self.layers[layer_index].self_attn
        query_heads = self.num_attention_heads
        key_value_heads = self.num_key_value_heads
        if query_heads % key_value_heads:
            raise ValueError(
                f"{query_heads} query heads are not divisible by {key_value_heads} KV heads"
            )
        return AttentionGeometry(
            num_query_heads=query_heads,
            num_key_value_heads=key_value_heads,
            num_key_value_groups=query_heads // key_value_heads,
            head_dim=int(getattr(attention, "head_dim", self.head_dim)),
            scaling=float(
                getattr(attention, "scaling", getattr(attention, "head_dim", self.head_dim) ** -0.5)
            ),
            sliding_window=getattr(attention, "sliding_window", None),
        )

    def capture(
        self,
        **kwargs: Any,
    ) -> ModelStateCapture:
        return ModelStateCapture(self, **kwargs)

    def run_with_capture(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        output_attentions: bool = False,
        **capture_kwargs: Any,
    ) -> tuple[Any, CapturedModelStates]:
        """Run the bare decoder once, avoiding materialization of vocabulary logits."""

        capture = self.capture(**capture_kwargs)
        with torch.inference_mode(), capture:
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=False,
                return_dict=True,
            )
        return outputs, capture.state

    def normalize_for_logit_lens(self, hidden: Tensor) -> Tensor:
        norm_device = _module_device(self.final_norm)
        return self.final_norm(hidden.to(norm_device))

    def project_logits(self, normalized_hidden: Tensor, start: int, end: int) -> Tensor:
        """Project a vocabulary slice without constructing full-vocabulary logits."""

        if not 0 <= start < end <= self.vocab_size:
            raise ValueError(f"Invalid vocabulary slice [{start}, {end})")
        weight = self.lm_head.weight[start:end]
        bias = getattr(self.lm_head, "bias", None)
        if bias is not None:
            bias = bias[start:end]
        hidden = normalized_hidden.to(device=weight.device, dtype=weight.dtype)
        return torch.nn.functional.linear(hidden, weight, bias)

    @torch.inference_mode()
    def recompute_attention(
        self,
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
        """Recompute selected QK attention entries for Llama or Qwen3.

        ``query_inputs`` and ``key_inputs`` must be the normalized hidden states
        received by ``self_attn`` (captured by :class:`ModelStateCapture`).  When
        ``normalize`` is true, softmax is over the supplied key set.  Supplying
        every input key reproduces HF eager attention exactly; supplying only
        retrieved-context keys gives the same top-k ordering and a
        context-renormalized distribution.
        """

        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"Invalid layer index {layer_index}")
        geometry = self.attention_geometry(layer_index)
        heads = tuple(int(head) for head in head_indices)
        if not heads:
            raise ValueError("head_indices cannot be empty")
        if len(set(heads)) != len(heads):
            raise ValueError("head_indices cannot contain duplicates")
        if any(not 0 <= head < geometry.num_query_heads for head in heads):
            raise IndexError(f"Invalid query-head indices: {heads}")
        if query_inputs.ndim != 3 or key_inputs.ndim != 3:
            raise ValueError("query_inputs and key_inputs must have shape [batch, tokens, hidden]")
        if query_inputs.shape[0] != key_inputs.shape[0]:
            raise ValueError("query_inputs and key_inputs must have equal batch sizes")
        if query_inputs.shape[-1] != self.hidden_size or key_inputs.shape[-1] != self.hidden_size:
            raise ValueError("Attention inputs do not match the model hidden size")

        attention = self.layers[layer_index].self_attn
        projection_device = _module_device(attention.q_proj)
        query_inputs = query_inputs.to(projection_device)
        key_inputs = key_inputs.to(projection_device)
        batch_size, query_length, _ = query_inputs.shape
        key_length = key_inputs.shape[1]

        query = attention.q_proj(query_inputs).view(
            batch_size,
            query_length,
            geometry.num_query_heads,
            geometry.head_dim,
        )
        key = attention.k_proj(key_inputs).view(
            batch_size,
            key_length,
            geometry.num_key_value_heads,
            geometry.head_dim,
        )
        if hasattr(attention, "q_norm"):
            query = attention.q_norm(query)
        if hasattr(attention, "k_norm"):
            key = attention.k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)

        query_position_tensor = _expand_absolute_positions(
            query_positions,
            batch_size=batch_size,
            expected_length=query_length,
            device=projection_device,
            name="query_positions",
        )
        key_position_tensor = _expand_absolute_positions(
            key_positions,
            batch_size=batch_size,
            expected_length=key_length,
            device=projection_device,
            name="key_positions",
        )
        query_cos, query_sin = self.decoder.rotary_emb(
            query_inputs,
            query_position_tensor,
        )
        key_cos, key_sin = self.decoder.rotary_emb(
            key_inputs,
            key_position_tensor,
        )
        query = _apply_rotary(query, query_cos, query_sin)
        key = _apply_rotary(key, key_cos, key_sin)

        head_index = torch.tensor(heads, dtype=torch.long, device=projection_device)
        kv_index = torch.div(
            head_index,
            geometry.num_key_value_groups,
            rounding_mode="floor",
        )
        selected_query = query.index_select(1, head_index)
        selected_key = key.index_select(1, kv_index)
        logits = torch.matmul(
            selected_query.float(),
            selected_key.float().transpose(-2, -1),
        )
        logits.mul_(geometry.scaling)

        valid = key_position_tensor[:, None, None, :] <= query_position_tensor[:, None, :, None]
        if geometry.sliding_window is not None:
            valid &= key_position_tensor[:, None, None, :] > (
                query_position_tensor[:, None, :, None] - int(geometry.sliding_window)
            )
        if key_attention_mask is not None:
            if key_attention_mask.shape != (batch_size, key_length):
                raise ValueError(
                    "key_attention_mask must have shape "
                    f"{(batch_size, key_length)}, got {tuple(key_attention_mask.shape)}"
                )
            valid &= key_attention_mask.to(device=projection_device, dtype=torch.bool)[
                :, None, None, :
            ]
        if not torch.all(valid.any(dim=-1)):
            raise ValueError("At least one query has no valid key after causal/padding masking")
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1) if normalize else logits


def _expand_absolute_positions(
    positions: Sequence[int] | Tensor,
    *,
    batch_size: int,
    expected_length: int,
    device: torch.device,
    name: str,
) -> Tensor:
    tensor = torch.as_tensor(positions, dtype=torch.long, device=device)
    if tensor.ndim == 1:
        if tensor.numel() != expected_length:
            raise ValueError(f"{name} has {tensor.numel()} entries; expected {expected_length}")
        tensor = tensor.unsqueeze(0).expand(batch_size, -1)
    elif tensor.ndim == 2:
        if tuple(tensor.shape) != (batch_size, expected_length):
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}; "
                f"expected {(batch_size, expected_length)}"
            )
    else:
        raise ValueError(f"{name} must be one- or two-dimensional")
    return tensor


def _rotate_half(hidden: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return hidden * cos + _rotate_half(hidden) * sin


def capture_model_states(
    adapter: DecoderModelAdapter,
    *,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
    **capture_kwargs: Any,
) -> CapturedModelStates:
    """Convenience wrapper returning only states from ``run_with_capture``."""

    _, state = adapter.run_with_capture(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        **capture_kwargs,
    )
    return state


def eager_attention_parity(
    adapter: DecoderModelAdapter,
    *,
    input_ids: Tensor,
    layer_index: int,
    head_indices: Sequence[int],
    query_positions: Sequence[int],
    key_positions: Sequence[int],
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
    atol: float = 2e-5,
    rtol: float = 2e-4,
) -> AttentionParityReport:
    """Compare manual QK reconstruction against Transformers eager attention.

    This is intended for tiny smoke tests because HF materializes full
    ``[batch, heads, sequence, sequence]`` attention tensors.
    """

    implementation = getattr(adapter.config, "_attn_implementation", None)
    if implementation != "eager":
        raise ValueError(
            "eager_attention_parity requires a model loaded with "
            "attn_implementation='eager'; "
            f"got {implementation!r}"
        )
    sequence_length = input_ids.shape[1]
    all_positions = tuple(range(sequence_length))
    outputs, captured = adapter.run_with_capture(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_attentions=True,
        ffn_positions=(),
        attention_layers=(layer_index,),
        attention_query_positions=query_positions,
        attention_key_positions=all_positions,
        last_hidden_positions=(),
    )
    if getattr(outputs, "attentions", None) is None:
        raise RuntimeError("The decoder did not return eager attention weights")

    absolute_positions: Tensor | Sequence[int]
    if position_ids is None:
        absolute_positions = all_positions
        query_absolute: Tensor | Sequence[int] = tuple(query_positions)
    else:
        absolute_positions = position_ids
        query_index = torch.tensor(query_positions, dtype=torch.long, device=position_ids.device)
        query_absolute = position_ids.index_select(1, query_index)

    full_recomputed = adapter.recompute_attention(
        layer_index=layer_index,
        query_inputs=captured.attention_queries[layer_index],
        key_inputs=captured.attention_keys[layer_index],
        query_positions=query_absolute,
        key_positions=absolute_positions,
        head_indices=head_indices,
        key_attention_mask=attention_mask,
    )
    key_index = torch.tensor(key_positions, dtype=torch.long, device=full_recomputed.device)
    recomputed = full_recomputed.index_select(-1, key_index)

    eager = outputs.attentions[layer_index]
    head_index = torch.tensor(head_indices, dtype=torch.long, device=eager.device)
    query_index = torch.tensor(query_positions, dtype=torch.long, device=eager.device)
    eager_key_index = torch.tensor(key_positions, dtype=torch.long, device=eager.device)
    eager = eager.index_select(1, head_index).index_select(2, query_index)
    eager = eager.index_select(3, eager_key_index).float()
    recomputed = recomputed.to(eager.device).float()
    absolute_error = (recomputed - eager).abs()
    denominator = eager.abs().clamp_min(torch.finfo(eager.dtype).eps)
    relative_error = absolute_error / denominator
    return AttentionParityReport(
        allclose=bool(torch.allclose(recomputed, eager, atol=atol, rtol=rtol)),
        max_absolute_error=float(absolute_error.max().item()),
        max_relative_error=float(relative_error.max().item()),
        recomputed=recomputed.detach().cpu(),
        eager=eager.detach().cpu(),
    )


def load_model_adapter(
    model_path: str,
    *,
    dtype: str | torch.dtype = "bfloat16",
    device_map: str | Mapping[str, Any] | None = "auto",
    attn_implementation: str = "sdpa",
    **from_pretrained_kwargs: Any,
) -> DecoderModelAdapter:
    """Lazily load a supported causal LM and return its adapter."""

    from transformers import AutoModelForCausalLM

    if isinstance(dtype, str):
        dtype_name = dtype.removeprefix("torch.")
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unknown torch dtype {dtype!r}")
        dtype = getattr(torch, dtype_name)
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
        **from_pretrained_kwargs,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return DecoderModelAdapter.from_model(model)
