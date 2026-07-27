"""Load and apply the exact released ReDeEP Llama-3 token configuration.

The official repository stores the selected Copying Heads, Knowledge FFNs,
normalization ranges, and score weight in a JSON artifact.  This module treats
that artifact as frozen configuration.  It does not inspect labels, rank
features, or fit new normalization ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


OFFICIAL_REPOSITORY = "https://github.com/Jeryi-Sun/ReDEeP-ICLR"
OFFICIAL_UPSTREAM_COMMIT = "4d081915b8fb4430fda65c411da61540cc73cc57"
OFFICIAL_CONFIG_RELATIVE_PATH = Path(
    "ReDeEP/log/test_llama3_8B/token_hyperparameter.json"
)
OFFICIAL_CONFIG_SHA256 = (
    "49a5b7f240432a0ddc777a6a6d3dff3053083762007d29f8401236adba77cc39"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _max_min(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be the released [max, min] pair")
    maximum, minimum = float(value[0]), float(value[1])
    if maximum <= minimum:
        raise ValueError(f"{name} has an invalid [max, min] order")
    return maximum, minimum


@dataclass(frozen=True)
class ReleasedTokenConfiguration:
    """Frozen values published for ReDeEP(Token), Llama-3-8B, RAGTruth."""

    selected_heads: tuple[tuple[int, int], ...]
    selected_layers: tuple[int, ...]
    head_max_min: tuple[float, float]
    layers_max_min: tuple[float, float]
    final_max_min: tuple[float, float]
    beta: float
    source_file_sha256: str
    target_architecture: str

    @property
    def cross_architecture_transfer(self) -> bool:
        return self.target_architecture != "llama"

    def manifest(self) -> dict[str, Any]:
        return {
            "configuration_mode": "frozen_released",
            "source_repository": OFFICIAL_REPOSITORY,
            "source_commit": OFFICIAL_UPSTREAM_COMMIT,
            "source_file": OFFICIAL_CONFIG_RELATIVE_PATH.as_posix(),
            "source_file_sha256": self.source_file_sha256,
            "source_architecture": "llama",
            "target_architecture": self.target_architecture,
            "cross_architecture_transfer": self.cross_architecture_transfer,
            "selected_heads": [list(head) for head in self.selected_heads],
            "selected_layers": list(self.selected_layers),
            "head_max_min": list(self.head_max_min),
            "layers_max_min": list(self.layers_max_min),
            "final_max_min": list(self.final_max_min),
            "beta": self.beta,
        }


def load_released_llama3_token_config(
    repository_root: str | Path,
    *,
    target_architecture: str = "llama",
    allow_cross_architecture_transfer: bool = False,
) -> ReleasedTokenConfiguration:
    """Read the official artifact without recalibration or label access."""

    architecture = str(target_architecture).lower()
    if architecture not in {"llama", "qwen3"}:
        raise ValueError(f"unsupported target architecture {architecture!r}")
    if architecture != "llama" and not allow_cross_architecture_transfer:
        raise ValueError(
            "cross-architecture transfer of the released Llama configuration "
            "requires explicit opt-in"
        )

    path = Path(repository_root) / OFFICIAL_CONFIG_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = _sha256(path)
    if actual_sha256 != OFFICIAL_CONFIG_SHA256:
        raise ValueError(
            f"{path} has SHA256 {actual_sha256}, expected the released "
            f"artifact SHA256 {OFFICIAL_CONFIG_SHA256}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")

    required = {
        "select_heads",
        "select_layers",
        "head_max_min",
        "layers_max_min",
        "final_max_min",
        "weight",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{path} is missing released fields {missing}")

    selected_heads: list[tuple[int, int]] = []
    for index, head in enumerate(value["select_heads"]):
        if (
            not isinstance(head, list)
            or len(head) != 2
            or any(
                isinstance(component, bool) or not isinstance(component, int)
                for component in head
            )
        ):
            raise ValueError(f"released head {index} is invalid")
        selected_heads.append((int(head[0]), int(head[1])))
    selected_layers = tuple(int(layer) for layer in value["select_layers"])
    if len(set(selected_heads)) != len(selected_heads):
        raise ValueError("released selected heads contain duplicates")
    if len(set(selected_layers)) != len(selected_layers):
        raise ValueError("released selected layers contain duplicates")

    beta = float(value["weight"])
    if beta <= 0.0:
        raise ValueError("released beta must be positive")
    return ReleasedTokenConfiguration(
        selected_heads=tuple(selected_heads),
        selected_layers=selected_layers,
        head_max_min=_max_min(value["head_max_min"], "head_max_min"),
        layers_max_min=_max_min(
            value["layers_max_min"], "layers_max_min"
        ),
        final_max_min=_max_min(value["final_max_min"], "final_max_min"),
        beta=beta,
        source_file_sha256=actual_sha256,
        target_architecture=architecture,
    )


def _released_normalize(value: float, bounds: Sequence[float]) -> float:
    maximum, minimum = float(bounds[0]), float(bounds[1])
    if maximum <= minimum:
        raise ValueError("released normalization range is invalid")
    return (float(value) - minimum) / (maximum - minimum)


def score_rows_with_released_config(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_heads: Sequence[Sequence[int]],
    config: ReleasedTokenConfiguration,
) -> list[dict[str, Any]]:
    """Apply the released score to already extracted ECS and PKS features."""

    normalized_feature_heads = [
        (int(head[0]), int(head[1])) for head in feature_heads
    ]
    head_to_index = {
        head: index for index, head in enumerate(normalized_feature_heads)
    }
    missing_heads = [
        head for head in config.selected_heads if head not in head_to_index
    ]
    if missing_heads:
        raise ValueError(
            f"features do not contain every released selected head: "
            f"{missing_heads}"
        )
    selected_head_indices = [
        head_to_index[head] for head in config.selected_heads
    ]
    maximum_layer = max(config.selected_layers)

    scored: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        external_rows = row.get("external")
        parametric_rows = row.get("parametric")
        if (
            not isinstance(external_rows, Sequence)
            or not isinstance(parametric_rows, Sequence)
            or len(external_rows) != len(parametric_rows)
        ):
            raise ValueError(
                f"row {row_index} has inconsistent released features"
            )
        token_scores: list[float] = []
        for token_index, (external, parametric) in enumerate(
            zip(external_rows, parametric_rows)
        ):
            if len(external) != len(normalized_feature_heads):
                raise ValueError(
                    f"row {row_index} token {token_index} has the wrong "
                    "ECS dimension"
                )
            if len(parametric) <= maximum_layer:
                raise ValueError(
                    f"row {row_index} token {token_index} has the wrong "
                    "PKS dimension"
                )
            external_sum = sum(
                float(external[index]) for index in selected_head_indices
            )
            parametric_sum = sum(
                float(parametric[layer]) for layer in config.selected_layers
            )
            token_scores.append(
                _released_normalize(
                    parametric_sum, config.layers_max_min
                )
                - config.beta
                * _released_normalize(
                    external_sum, config.head_max_min
                )
            )
        scored.append({**row, "scores": token_scores})
    return scored
