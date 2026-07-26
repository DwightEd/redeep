from __future__ import annotations

# ruff: noqa: E402, I001

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from redeep.models import (
    DecoderModelAdapter,
    capture_model_states,
    eager_attention_parity,
)


def _rotate_half(hidden):
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class TinyRotary(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, hidden, position_ids):
        frequencies = torch.arange(
            1,
            self.head_dim // 2 + 1,
            dtype=torch.float32,
            device=hidden.device,
        )
        angles = position_ids.float().unsqueeze(-1) * frequencies / 17.0
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden):
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden.float() * torch.rsqrt(variance + 1e-6)
        return (normalized * self.weight.float()).to(hidden.dtype)


class TinyAttention(nn.Module):
    def __init__(self, config, *, qk_norm: bool):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.sliding_window = None
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        if qk_norm:
            self.q_norm = TinyRMSNorm(self.head_dim)
            self.k_norm = TinyRMSNorm(self.head_dim)

    def forward(
        self,
        *,
        hidden_states,
        attention_mask,
        position_embeddings,
        output_attentions=False,
        **_kwargs,
    ):
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch,
            length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        key = self.k_proj(hidden_states).view(
            batch,
            length,
            self.config.num_key_value_heads,
            self.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            batch,
            length,
            self.config.num_key_value_heads,
            self.head_dim,
        )
        if hasattr(self, "q_norm"):
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = position_embeddings
        query = query * cos[:, None] + _rotate_half(query) * sin[:, None]
        key = key * cos[:, None] + _rotate_half(key) * sin[:, None]
        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        logits = query.float() @ key.float().transpose(-2, -1)
        logits *= self.scaling
        if attention_mask is not None:
            logits += attention_mask
        weights = logits.softmax(dim=-1)
        output = weights.to(value.dtype) @ value
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output), weights if output_attentions else None


class TinyLayer(nn.Module):
    def __init__(self, config, *, qk_norm: bool):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(config.hidden_size)
        self.self_attn = TinyAttention(config, qk_norm=qk_norm)
        self.post_attention_layernorm = TinyRMSNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False),
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        output_attentions=False,
        **kwargs,
    ):
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attention, weights = self.self_attn(
            hidden_states=normalized,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )
        before_ffn = residual + attention
        output = before_ffn + self.mlp(self.post_attention_layernorm(before_ffn))
        return (output, weights) if output_attentions else (output,)


class TinyDecoder(nn.Module):
    def __init__(self, config, *, qk_norm: bool):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TinyLayer(config, qk_norm=qk_norm) for _ in range(config.num_hidden_layers)]
        )
        self.norm = TinyRMSNorm(config.hidden_size)
        self.rotary_emb = TinyRotary(config.head_dim)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        output_attentions=False,
        **_kwargs,
    ):
        hidden = self.embed_tokens(input_ids)
        batch, length = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(length, device=input_ids.device)[None]
        cos, sin = self.rotary_emb(hidden, position_ids)
        causal = torch.full(
            (batch, 1, length, length),
            torch.finfo(torch.float32).min,
            device=input_ids.device,
        )
        causal = torch.triu(causal, diagonal=1)
        if attention_mask is not None:
            causal = causal.masked_fill(
                ~attention_mask[:, None, None, :].bool(),
                torch.finfo(torch.float32).min,
            )
        attentions = []
        for layer in self.layers:
            outputs = layer(
                hidden,
                attention_mask=causal,
                position_embeddings=(cos, sin),
                output_attentions=output_attentions,
            )
            hidden = outputs[0]
            if output_attentions:
                attentions.append(outputs[1])
        return SimpleNamespace(
            last_hidden_state=self.norm(hidden),
            attentions=tuple(attentions) if output_attentions else None,
        )


class TinyCausalLM(nn.Module):
    def __init__(self, *, model_type="llama", qk_norm=False):
        super().__init__()
        self.config = SimpleNamespace(
            model_type=model_type,
            hidden_size=8,
            head_dim=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            vocab_size=13,
            _attn_implementation="eager",
        )
        self.model = TinyDecoder(self.config, qk_norm=qk_norm)
        self.lm_head = nn.Linear(8, 13, bias=False)

    def get_decoder(self):
        return self.model

    def get_output_embeddings(self):
        return self.lm_head


@pytest.fixture
def llama_adapter():
    torch.manual_seed(4)
    return DecoderModelAdapter.from_model(TinyCausalLM())


def test_adapter_geometry_and_residual_hooks(llama_adapter):
    adapter = llama_adapter
    assert adapter.num_layers == 2
    assert adapter.num_attention_heads == 4
    assert adapter.num_key_value_heads == 2
    assert adapter.attention_geometry(0).num_key_value_groups == 2

    input_ids = torch.tensor([[1, 2, 3, 4]])
    state = capture_model_states(
        adapter,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        ffn_positions=(1, 3),
        attention_layers=(0,),
        attention_query_positions=(2, 3),
        attention_key_positions=(0, 1, 2, 3),
        last_hidden_positions=(0, 2),
    )
    assert set(state.pre_ffn) == {0, 1}
    assert set(state.post_ffn) == {0, 1}
    assert state.pre_ffn[0].shape == (1, 2, 8)
    assert state.attention_queries[0].shape == (1, 2, 8)
    assert state.attention_keys[0].shape == (1, 4, 8)
    assert state.last_hidden.shape == (1, 2, 8)

    expected_delta = adapter.layers[0].mlp(
        adapter.layers[0].post_attention_layernorm(state.pre_ffn[0])
    )
    torch.testing.assert_close(
        state.post_ffn[0] - state.pre_ffn[0],
        expected_delta,
    )


@pytest.mark.parametrize(
    ("model_type", "qk_norm"),
    [("llama", False), ("qwen3", True)],
)
def test_selected_attention_matches_tiny_eager(model_type, qk_norm):
    torch.manual_seed(7)
    adapter = DecoderModelAdapter.from_model(
        TinyCausalLM(model_type=model_type, qk_norm=qk_norm)
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    report = eager_attention_parity(
        adapter,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        layer_index=1,
        head_indices=(0, 3),
        query_positions=(2, 4),
        key_positions=(0, 1, 2, 3, 4),
        atol=1e-6,
        rtol=1e-5,
    )
    assert report.allclose, report
    assert report.max_absolute_error < 1e-6


def test_parity_requires_explicit_eager_model(llama_adapter):
    llama_adapter.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="attn_implementation='eager'"):
        eager_attention_parity(
            llama_adapter,
            input_ids=torch.tensor([[1, 2]]),
            layer_index=0,
            head_indices=(0,),
            query_positions=(1,),
            key_positions=(0, 1),
        )
