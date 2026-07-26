"""Auditable Copying Head discovery for large-vocabulary GQA models.

The ReDeEP appendix ranks heads by (1) the absolute trace of the vocabulary OV
circuit and (2) an IQR outlier count over Gershgorin boundary points.  Building
the full vocabulary-by-vocabulary circuit is infeasible for the target models.
Here the trace is still exact (up to float32 arithmetic) through a low-rank
cyclic identity, while Gershgorin radii are explicitly marked as a
deterministic sampled approximation.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from redeep.models import DecoderModelAdapter


@dataclass(frozen=True)
class GershgorinEstimate:
    sampled_outlier_count: int
    sampled_outlier_fraction: float
    first_quartile: float
    third_quartile: float
    iqr: float
    lower_fence: float
    upper_fence: float
    mean_estimated_radius: float
    maximum_estimated_radius: float


@dataclass(frozen=True)
class CopyHeadRecord:
    layer: int
    head: int
    kv_head: int
    trace: float
    absolute_trace: float
    sampled_outlier_count: int
    sampled_outlier_fraction: float
    gershgorin_q1: float
    gershgorin_q3: float
    gershgorin_iqr: float
    gershgorin_lower_fence: float
    gershgorin_upper_fence: float
    mean_estimated_radius: float
    maximum_estimated_radius: float
    trace_rank: int
    outlier_rank: int
    rank_sum: int
    copying_rank: int

    @property
    def copying_score(self) -> float:
        """Monotone score where larger means a stronger copying candidate."""

        return -float(self.rank_sum)


@dataclass(frozen=True)
class CopyHeadDiscovery:
    records: tuple[CopyHeadRecord, ...]
    selected_heads: tuple[tuple[int, int], ...]
    metadata: dict[str, Any]

    @property
    def top_heads(self) -> tuple[tuple[int, int], ...]:
        return self.selected_heads

    def to_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record in self.records:
            row = asdict(record)
            row["copying_score"] = record.copying_score
            rows.append(row)
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.to_rows(),
            "selected_heads": [list(head) for head in self.selected_heads],
            "metadata": self.metadata,
        }


def _compute_device(
    adapter: DecoderModelAdapter,
    compute_device: str | torch.device | None,
) -> torch.device:
    device = adapter.output_device if compute_device is None else torch.device(compute_device)
    if device.type == "meta":
        raise ValueError("Copying-head discovery cannot run with a meta compute device")
    return device


def compute_vocab_gram(
    adapter: DecoderModelAdapter,
    *,
    vocab_block_size: int = 4096,
    compute_device: str | torch.device | None = None,
    output_device: str | torch.device | None = None,
) -> Tensor:
    """Compute ``W_U.T @ W_E`` exactly without a vocabulary-square matrix.

    This is the expensive model-level term, so callers may cache the returned
    ``[hidden, hidden]`` tensor and pass it back to :func:`discover_copy_heads`.
    Accumulation is deterministic with a fixed device/library stack and block
    size; float32 is used even when model weights are BF16.
    """

    if vocab_block_size <= 0:
        raise ValueError("vocab_block_size must be positive")
    device = _compute_device(adapter, compute_device)
    embedding = adapter.embed_tokens.weight
    unembedding = adapter.lm_head.weight
    vocabulary_size = min(embedding.shape[0], unembedding.shape[0])
    if embedding.shape[1] != adapter.hidden_size or unembedding.shape[1] != adapter.hidden_size:
        raise ValueError("Embedding/unembedding hidden dimensions do not match the adapter")
    gram = torch.zeros(
        (adapter.hidden_size, adapter.hidden_size),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        for start in range(0, vocabulary_size, vocab_block_size):
            end = min(start + vocab_block_size, vocabulary_size)
            embedding_block = embedding[start:end].to(device=device, dtype=torch.float32)
            unembedding_block = unembedding[start:end].to(device=device, dtype=torch.float32)
            gram.add_(unembedding_block.transpose(0, 1) @ embedding_block)
    if output_device is not None:
        gram = gram.to(output_device)
    return gram


def exact_low_rank_trace(
    gram: Tensor,
    value_head_weight: Tensor,
    output_head_weight: Tensor,
) -> Tensor:
    """Exact trace of ``W_E W_V^T W_O^T W_U^T`` via cyclic permutation.

    Args:
        gram: ``W_U.T @ W_E`` with shape ``[hidden, hidden]``.
        value_head_weight: PyTorch V-projection block ``[head_dim, hidden]``.
        output_head_weight: PyTorch O-projection column block
            ``[hidden, head_dim]``.
    """

    hidden_size = gram.shape[0]
    if gram.shape != (hidden_size, hidden_size):
        raise ValueError("gram must be square")
    if value_head_weight.ndim != 2 or value_head_weight.shape[1] != hidden_size:
        raise ValueError("value_head_weight must have shape [head_dim, hidden]")
    if output_head_weight.shape != (hidden_size, value_head_weight.shape[0]):
        raise ValueError("output_head_weight must have shape [hidden, head_dim]")
    device = gram.device
    value = value_head_weight.to(device=device, dtype=gram.dtype)
    output = output_head_weight.to(device=device, dtype=gram.dtype)
    gram_value = gram @ value.transpose(0, 1)
    return (gram_value * output).sum()


def _sample_indices(vocabulary_size: int, sample_size: int, seed: int) -> tuple[int, ...]:
    if sample_size < 2:
        raise ValueError("gershgorin_sample_size must be at least 2")
    sample_size = min(sample_size, vocabulary_size)
    generator = random.Random(int(seed))
    return tuple(sorted(generator.sample(range(vocabulary_size), sample_size)))


def estimate_sampled_gershgorin(
    sampled_embeddings: Tensor,
    sampled_unembeddings: Tensor,
    value_head_weight: Tensor,
    output_head_weight: Tensor,
    *,
    vocabulary_size: int,
    row_block_size: int = 128,
) -> GershgorinEstimate:
    """Estimate full-row Gershgorin radii from deterministic sampled columns.

    The same vocabulary indices are used for rows and columns, preserving an
    exact diagonal for every sampled row.  The mean sampled off-diagonal
    magnitude is multiplied by ``vocabulary_size - 1``.  Consequently these
    disks are estimates, not rigorous Gershgorin bounds.
    """

    if sampled_embeddings.shape != sampled_unembeddings.shape:
        raise ValueError("Sampled embedding and unembedding tensors must align")
    if sampled_embeddings.ndim != 2:
        raise ValueError("Sampled weights must have shape [sample, hidden]")
    sample_size, hidden_size = sampled_embeddings.shape
    if sample_size < 2:
        raise ValueError("At least two vocabulary samples are required")
    if vocabulary_size < sample_size:
        raise ValueError("vocabulary_size cannot be smaller than the sample")
    if row_block_size <= 0:
        raise ValueError("row_block_size must be positive")
    if value_head_weight.shape[1] != hidden_size:
        raise ValueError("V head does not match the hidden dimension")
    if output_head_weight.shape != (hidden_size, value_head_weight.shape[0]):
        raise ValueError("O head does not match V head dimensions")

    device = value_head_weight.device
    embeddings = sampled_embeddings.to(device=device, dtype=torch.float32)
    unembeddings = sampled_unembeddings.to(device=device, dtype=torch.float32)
    value = value_head_weight.to(device=device, dtype=torch.float32)
    output = output_head_weight.to(device=device, dtype=torch.float32)
    left = embeddings @ value.transpose(0, 1)
    right = unembeddings @ output
    boundaries = torch.empty(sample_size * 2, dtype=torch.float32, device=device)
    radii = torch.empty(sample_size, dtype=torch.float32, device=device)
    radius_scale = (vocabulary_size - 1) / (sample_size - 1)

    with torch.inference_mode():
        for start in range(0, sample_size, row_block_size):
            end = min(start + row_block_size, sample_size)
            sampled_rows = left[start:end] @ right.transpose(0, 1)
            center = (left[start:end] * right[start:end]).sum(dim=-1)
            sampled_off_diagonal = sampled_rows.abs().sum(dim=-1) - center.abs()
            radius = sampled_off_diagonal.clamp_min(0.0) * radius_scale
            radii[start:end] = radius
            boundaries[2 * start : 2 * end : 2] = center - radius
            boundaries[2 * start + 1 : 2 * end : 2] = center + radius

        q1, q3 = torch.quantile(
            boundaries,
            torch.tensor([0.25, 0.75], device=device),
        )
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((boundaries < lower) | (boundaries > upper)).sum().item())
    return GershgorinEstimate(
        sampled_outlier_count=outlier_count,
        sampled_outlier_fraction=outlier_count / boundaries.numel(),
        first_quartile=float(q1.item()),
        third_quartile=float(q3.item()),
        iqr=float(iqr.item()),
        lower_fence=float(lower.item()),
        upper_fence=float(upper.item()),
        mean_estimated_radius=float(radii.mean().item()),
        maximum_estimated_radius=float(radii.max().item()),
    )


def discover_copy_heads(
    adapter: DecoderModelAdapter,
    *,
    top_k: int = 32,
    vocab_block_size: int = 4096,
    gershgorin_sample_size: int = 2048,
    gershgorin_block_size: int = 128,
    seed: int = 2024,
    gram: Tensor | None = None,
    compute_device: str | torch.device | None = None,
) -> CopyHeadDiscovery:
    """Rank all query heads using the ReDeEP Appendix-B recipe.

    GQA is handled by mapping query head ``h`` to KV head
    ``h // num_key_value_groups``.  The head-specific O-projection block remains
    distinct, so all query heads receive their own score.
    """

    total_heads = adapter.num_layers * adapter.num_attention_heads
    if not 1 <= top_k <= total_heads:
        raise ValueError(f"top_k must be in [1, {total_heads}]")
    if gershgorin_block_size <= 0:
        raise ValueError("gershgorin_block_size must be positive")
    device = _compute_device(adapter, compute_device)
    if gram is None:
        gram = compute_vocab_gram(
            adapter,
            vocab_block_size=vocab_block_size,
            compute_device=device,
        )
    else:
        if gram.shape != (adapter.hidden_size, adapter.hidden_size):
            raise ValueError(
                f"gram must have shape {(adapter.hidden_size, adapter.hidden_size)}"
            )
        gram = gram.to(device=device, dtype=torch.float32)

    embedding = adapter.embed_tokens.weight
    unembedding = adapter.lm_head.weight
    vocabulary_size = min(embedding.shape[0], unembedding.shape[0])
    sample_indices = _sample_indices(vocabulary_size, gershgorin_sample_size, seed)
    sample_index_tensor = torch.tensor(sample_indices, dtype=torch.long, device=embedding.device)
    sampled_embeddings = embedding.index_select(0, sample_index_tensor).to(
        device=device,
        dtype=torch.float32,
    )
    unembedding_index = sample_index_tensor.to(unembedding.device)
    sampled_unembeddings = unembedding.index_select(0, unembedding_index).to(
        device=device,
        dtype=torch.float32,
    )

    raw_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for layer_index, layer in enumerate(adapter.layers):
            attention = layer.self_attn
            geometry = adapter.attention_geometry(layer_index)
            v_weight = attention.v_proj.weight
            o_weight = attention.o_proj.weight
            value_by_kv_head: dict[int, Tensor] = {}
            gram_value_by_kv_head: dict[int, Tensor] = {}
            sampled_left_by_kv_head: dict[int, Tensor] = {}
            for kv_head in range(geometry.num_key_value_heads):
                start = kv_head * geometry.head_dim
                end = start + geometry.head_dim
                value = v_weight[start:end].to(device=device, dtype=torch.float32)
                value_by_kv_head[kv_head] = value
                gram_value_by_kv_head[kv_head] = gram @ value.transpose(0, 1)
                sampled_left_by_kv_head[kv_head] = (
                    sampled_embeddings @ value.transpose(0, 1)
                )

            for head in range(geometry.num_query_heads):
                kv_head = head // geometry.num_key_value_groups
                output_start = head * geometry.head_dim
                output_end = output_start + geometry.head_dim
                output = o_weight[:, output_start:output_end].to(
                    device=device,
                    dtype=torch.float32,
                )
                trace = (gram_value_by_kv_head[kv_head] * output).sum()

                # Reuse the sampled E @ V term shared by each GQA group.
                left = sampled_left_by_kv_head[kv_head]
                right = sampled_unembeddings @ output
                boundaries = torch.empty(
                    len(sample_indices) * 2,
                    dtype=torch.float32,
                    device=device,
                )
                radii = torch.empty(
                    len(sample_indices),
                    dtype=torch.float32,
                    device=device,
                )
                radius_scale = (vocabulary_size - 1) / (len(sample_indices) - 1)
                for row_start in range(
                    0,
                    len(sample_indices),
                    gershgorin_block_size,
                ):
                    row_end = min(
                        row_start + gershgorin_block_size,
                        len(sample_indices),
                    )
                    sampled_rows = left[row_start:row_end] @ right.transpose(0, 1)
                    center = (left[row_start:row_end] * right[row_start:row_end]).sum(
                        dim=-1
                    )
                    off_diagonal = sampled_rows.abs().sum(dim=-1) - center.abs()
                    radius = off_diagonal.clamp_min(0.0) * radius_scale
                    radii[row_start:row_end] = radius
                    boundaries[2 * row_start : 2 * row_end : 2] = center - radius
                    boundaries[2 * row_start + 1 : 2 * row_end : 2] = center + radius
                q1, q3 = torch.quantile(
                    boundaries,
                    torch.tensor([0.25, 0.75], device=device),
                )
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(
                    ((boundaries < lower) | (boundaries > upper)).sum().item()
                )
                raw_records.append(
                    {
                        "layer": layer_index,
                        "head": head,
                        "kv_head": kv_head,
                        "trace": float(trace.item()),
                        "absolute_trace": float(trace.abs().item()),
                        "sampled_outlier_count": outlier_count,
                        "sampled_outlier_fraction": outlier_count
                        / boundaries.numel(),
                        "gershgorin_q1": float(q1.item()),
                        "gershgorin_q3": float(q3.item()),
                        "gershgorin_iqr": float(iqr.item()),
                        "gershgorin_lower_fence": float(lower.item()),
                        "gershgorin_upper_fence": float(upper.item()),
                        "mean_estimated_radius": float(radii.mean().item()),
                        "maximum_estimated_radius": float(radii.max().item()),
                    }
                )

    trace_order = sorted(
        raw_records,
        key=lambda row: (-row["absolute_trace"], row["layer"], row["head"]),
    )
    trace_ranks = {
        (row["layer"], row["head"]): rank
        for rank, row in enumerate(trace_order, start=1)
    }
    outlier_order = sorted(
        raw_records,
        key=lambda row: (
            row["sampled_outlier_count"],
            row["layer"],
            row["head"],
        ),
    )
    outlier_ranks = {
        (row["layer"], row["head"]): rank
        for rank, row in enumerate(outlier_order, start=1)
    }
    for row in raw_records:
        key = (row["layer"], row["head"])
        row["trace_rank"] = trace_ranks[key]
        row["outlier_rank"] = outlier_ranks[key]
        row["rank_sum"] = trace_ranks[key] + outlier_ranks[key]
    ranked_rows = sorted(
        raw_records,
        key=lambda row: (
            row["rank_sum"],
            -row["absolute_trace"],
            row["sampled_outlier_count"],
            row["layer"],
            row["head"],
        ),
    )
    records: list[CopyHeadRecord] = []
    for copying_rank, row in enumerate(ranked_rows, start=1):
        records.append(CopyHeadRecord(**row, copying_rank=copying_rank))

    sample_hash = hashlib.sha256(
        json.dumps(sample_indices, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    metadata: dict[str, Any] = {
        "method": "redeep_appendix_b_trace_plus_sampled_gershgorin_iqr",
        "matrix_convention": "M = W_E @ W_V.T @ W_O.T @ W_U.T",
        "trace_method": "exact cyclic trace via (W_U.T @ W_E) @ W_V.T",
        "trace_exact_up_to_float32": True,
        "gershgorin_exact": len(sample_indices) == vocabulary_size,
        "gershgorin_approximation": (
            "same deterministic sampled rows/columns; exact sampled diagonals; "
            "sample mean absolute off-diagonal scaled by vocab_size-1"
        ),
        "warning": (
            "When sample_size < vocab_size, estimated disks are not rigorous "
            "Gershgorin bounds and reproduce the paper recipe only approximately."
        ),
        "sample_seed": int(seed),
        "sample_size": len(sample_indices),
        "sample_indices": list(sample_indices),
        "sample_indices_sha256": sample_hash,
        "vocabulary_size": int(vocabulary_size),
        "vocab_block_size": int(vocab_block_size),
        "gershgorin_block_size": int(gershgorin_block_size),
        "radius_scale": (vocabulary_size - 1) / (len(sample_indices) - 1),
        "accumulation_dtype": "float32",
        "query_to_kv_mapping": "kv_head = query_head // num_key_value_groups",
        "bias_terms_included": False,
        "rank_recipe": (
            "ordinal rank(outlier_count ascending) + "
            "ordinal rank(abs(trace) descending); lower rank_sum is stronger"
        ),
        "tie_break": "layer ascending, then head ascending",
        "top_k": int(top_k),
    }
    selected_heads = tuple((record.layer, record.head) for record in records[:top_k])
    return CopyHeadDiscovery(
        records=tuple(records),
        selected_heads=selected_heads,
        metadata=metadata,
    )
