"""Shared typed records used by data, extraction, calibration, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, order=True)
class CharSpan:
    """A half-open character interval in the raw response."""

    start: int
    end: int
    text: str = field(default="", compare=False)
    label_type: str = field(default="", compare=False)
    implicit_true: bool = field(default=False, compare=False)
    due_to_null: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid half-open span [{self.start}, {self.end})")


@dataclass(frozen=True)
class RagTruthExample:
    """One RAGTruth response joined with its source record."""

    response_id: str
    source_id: str
    generator_model: str
    split: str
    quality: str
    task: str
    prompt: str
    context: str
    response: str
    spans: tuple[CharSpan, ...]
    source: str = ""
    source_info: Any = None

    @property
    def response_label(self) -> int:
        return int(bool(self.spans))


@dataclass(frozen=True)
class TokenizedExample:
    """A rendered, tokenized example with explicit causal-position mappings."""

    example: RagTruthExample
    rendered_text: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    response_token_positions: tuple[int, ...]
    predictor_positions: tuple[int, ...]
    response_offsets: tuple[tuple[int, int], ...]
    token_labels: tuple[int, ...]
    context_token_positions: tuple[int, ...]


@dataclass(frozen=True)
class FeatureManifest:
    """Metadata stored next to each feature shard."""

    schema_version: int
    model_name: str
    model_path: str
    dataset_hashes: dict[str, str]
    config_hash: str
    git_commit: str
    feature_mode: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
