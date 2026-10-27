"""``configs/bias_slices.yaml``: column mapping, slices, thresholds and the mitigation scheme."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..common.config import _load_model, project_root

SLICE_NAMES = (
    "state",
    "cms_region",
    "ownership_group",
    "bed_band",
    "chain_status",
    "in_hospital",
    "ccrc",
    "sff_status",
    "star_band",
)
# their source columns are not model features (feature_registry model_use), so they never feed training
NOT_WEIGHT_SLICES = ("star_band", "chain_status")
SCOPES = ("all_labeled", "train", "latest")
WEIGHT_SCHEMES = ("inverse_frequency", "balance_label_rate", "none")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DatasetCfg(_Model):
    table: str
    key: list[str]


class SlicesCfg(_Model):
    single: list[str]
    crossings: list[tuple[str, str]] = []
    # built from past inspections (SFF status, star rating), so a higher rate there is expected: reported in
    # their own section and never counted as problematic
    risk_tiers: list[str] = []

    @model_validator(mode="after")
    def _distinct_pairs(self) -> SlicesCfg:
        if same := [a for a, b in self.crossings if a == b]:
            raise ValueError(f"slices.crossings: a slice crossed with itself: {same}")
        return self


class Thresholds(_Model):
    min_slice_facilities: int = Field(ge=1)
    min_slice_rows: int = Field(ge=1)
    ratio_low: float = Field(gt=0)
    ratio_high: float = Field(gt=0)
    effect_size_T: float = Field(ge=0)
    fdr_alpha: float = Field(gt=0, lt=1)
    top_k: int = Field(ge=1)
    missingness_gap: float = Field(ge=0, le=1)
    skew_js: float = Field(ge=0, le=1)
    skew_linf: float = Field(ge=0, le=1)
    censoring_gap: float = Field(ge=0, le=1)
    ci_level: float = Field(gt=0, lt=1)


class SkewCfg(_Model):
    reference_scope: str
    comparison_scope: str
    split_label: str
    numeric_bins: int = Field(ge=2)
    max_features: int | None = Field(default=None, ge=1)
    null_draws: int = Field(ge=1)
    null_quantile: float = Field(gt=0, lt=1)
    seed: int = 0


class RaterBiasCfg(_Model):
    group_column: str
    exposure_columns: list[str]
    rate_column: str
    quantile: float = Field(gt=0, lt=1)
    min_state_facilities: int = Field(ge=1)


class MitigationCfg(_Model):
    weights: Literal["inverse_frequency", "balance_label_rate", "none"]
    weight_slice: str
    weight_label: str
    clip: tuple[float, float]

    @model_validator(mode="after")
    def _clip_order(self) -> MitigationCfg:
        lo, hi = self.clip
        if not 0 < lo <= 1 <= hi:
            raise ValueError("mitigation.clip must be [low, high] with 0 < low <= 1 <= high")
        return self


class BiasConfig(_Model):
    version: int
    dataset: DatasetCfg
    labels: list[str] = Field(min_length=1)
    status_column_pattern: str
    split_column_pattern: str
    labeled_values: list[str] = Field(min_length=2)
    columns: dict[str, str]
    slices: SlicesCfg
    bed_band_edges: list[float]
    bed_band_labels: list[str]
    cms_region_map: dict[str, int]
    thresholds: Thresholds
    skew: SkewCfg
    rater_bias: RaterBiasCfg
    mitigation: MitigationCfg
    output_dir: str

    @model_validator(mode="after")
    def _consistent(self) -> BiasConfig:
        unknown = sorted(
            {
                *self.slices.single,
                *(s for pair in self.slices.crossings for s in pair),
                *self.slices.risk_tiers,
            }
            - set(SLICE_NAMES)
        )
        if unknown:
            raise ValueError(f"slices: unknown slice names {unknown}; known: {list(SLICE_NAMES)}")
        if self.mitigation.weight_slice not in SLICE_NAMES:
            raise ValueError(
                f"mitigation.weight_slice: unknown slice {self.mitigation.weight_slice}"
            )
        if self.mitigation.weight_slice in NOT_WEIGHT_SLICES:
            raise ValueError(
                f"mitigation.weight_slice: {self.mitigation.weight_slice} reads a column that is not a model "
                f"feature (star ratings are ablation-only, chain_id product-only); not one of {list(NOT_WEIGHT_SLICES)}"
            )
        edges = self.bed_band_edges
        if len(edges) != len(self.bed_band_labels) + 1:
            raise ValueError("bed_band_edges must have one more entry than bed_band_labels")
        if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
            raise ValueError("bed_band_edges must be strictly increasing")
        for label in [*self.labels, self.skew.split_label, self.mitigation.weight_label]:
            if not label.startswith("y_"):
                raise ValueError(f"label {label} must be a y_<name> column")
        return self

    def status_column(self, label: str) -> str:
        """``y_g12`` -> ``status_g12`` (status_column_pattern)."""
        return self.status_column_pattern.format(label_suffix=label.removeprefix("y_"))

    def split_column(self, label: str) -> str:
        """``y_g12`` -> ``split_g12`` (split_column_pattern)."""
        return self.split_column_pattern.format(label_suffix=label.removeprefix("y_"))


def default_config_path() -> Path:
    return project_root() / "configs" / "bias_slices.yaml"


def load_bias_config(path: str | Path | None = None) -> BiasConfig:
    """Load and validate ``configs/bias_slices.yaml`` (ConfigError when it is missing or invalid)."""
    return _load_model(BiasConfig, path or default_config_path())
