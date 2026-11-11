"""The run folder and the JSON files the ML tasks hand each other (docs/ML_PIPELINE_SPEC.md §0.7, §1.2, §2, §4.5, §9).

One cms_nh_ml run writes under ``<root_prefix><run_prefix>/<ml_run_id>/`` (``RunPaths``). Each task reads what the
earlier ones wrote, never gold itself: ``prepare`` pins gold into ``input/`` and writes the frames. The models below
are the contracts between work packages; every file carries ``schema_version`` so a later reader can refuse a format
it doesn't know. ``evaluation.json`` belongs to W2 alone (written and read by ``evaluate`` and ``gate``), so it has no
model here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..common.config import MlConfig

Frame = Literal["eval_train", "test", "refit_train", "score", "prev_score"]
FRAMES: tuple[Frame, ...] = ("eval_train", "test", "refit_train", "score", "prev_score")
Candidate = Literal["ensemble", "lgbm", "lr"]
DecisionName = Literal["SKIP", "RETRAIN", "RETUNE", "SCORE", "PROMOTE", "REFUSE"]


@dataclass(frozen=True)
class RunPaths:
    """Object-store keys of one run (relative to the store root, so the same on local disk and GCS)."""

    root_prefix: str
    run_prefix: str
    ml_run_id: str

    @classmethod
    def of(cls, cfg: MlConfig, ml_run_id: str) -> RunPaths:
        return cls(cfg.outputs.root_prefix, cfg.outputs.run_prefix, ml_run_id)

    @property
    def gold_prefix(self) -> str:
        return f"{self.root_prefix}gold/"

    @property
    def run_dir(self) -> str:
        return f"{self.root_prefix}{self.run_prefix}/{self.ml_run_id}/"

    @property
    def input_dir(self) -> str:
        return f"{self.run_dir}input/"

    def input_file(self, name: str) -> str:
        """The pinned copy of gold's ``name`` (merged parquet, dataset_manifest.json, _current.json)."""
        return f"{self.input_dir}{name}"

    @property
    def input_manifest(self) -> str:
        return f"{self.input_dir}manifest.json"

    def frame(self, name: Frame) -> str:
        return f"{self.run_dir}frames/{name}.parquet"

    @property
    def windows(self) -> str:
        return f"{self.run_dir}windows.json"

    @property
    def decision(self) -> str:
        return f"{self.run_dir}decision.json"

    @property
    def champion(self) -> str:
        return f"{self.run_dir}champion.json"

    @property
    def selection(self) -> str:
        return f"{self.run_dir}selection.json"

    @property
    def evaluation(self) -> str:
        return f"{self.run_dir}evaluation.json"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1


class Decision(_Contract):
    """``decide``'s result (§2, §15.2): its last stdout line (the DAG branches on it) and ``decision.json``."""

    decision: DecisionName
    proceed: bool
    reason: str
    mode: Literal["auto", "retrain", "retune", "score_only", "promote"]
    gold_run_id: str | None = None
    v: str | None = None
    v_labels: str | None = None
    dataset_version: str | None = None
    champion_version: int | None = None


class MonthRange(_Contract):
    """Calendar months ``[first, last]`` (inclusive) and the months in it that hold rows."""

    first: str
    last: str
    months: list[str]


class Windows(_Contract):
    """``windows.json`` (§3.2): every window of one run, for one label."""

    label: str
    horizon_months: int
    buffer_months: int
    v: str
    v_labels: str
    last_labeled_t: str  # L = V_L - (H + B)
    eval_gap: Literal["production_lag", "purge"]
    train_months: int
    spine_start: str
    eval: MonthRange
    test: MonthRange
    refit: MonthRange
    score_month: str  # V
    prev_score_month: (
        str | None
    )  # the latest month with rows before V (score drift with the same model, §12.1)
    development_months: list[
        str
    ]  # test months chosen on by earlier work (§3.2), marked in every report


class LeakageFlag(_Contract):
    """A feature that alone ranks the label too well on a training frame (§1.5): holds promotion at G2."""

    frame: Frame
    feature: str
    pr_auc: float
    threshold: float
    cleared: bool  # listed, with a reason, in ml.yaml checks.leakage_cleared


class RegimeFlag(_Contract):
    """A regime-sensitive feature whose score-month values the refit window doesn't cover (§1.4)."""

    feature: str
    detail: str


class InputManifest(_Contract):
    """``input/manifest.json``: what ``prepare`` pinned and built (§1.2, §2). Written last, it names the hash of every
    other file of the run's input, so later tasks trust it and nothing else."""

    ml_run_id: str
    gold_run_id: str
    v: str
    v_labels: str
    dataset_version: str
    verified: bool
    merged_sha256: str
    gold_git_commit: str | None
    gold_config_sha256: dict[str, str]
    flags: dict[str, Any]  # gold's manifest flags, e.g. stale_refresh months
    label: str
    features: list[str]  # the model input, in order (registry model list minus features.exclude)
    excluded_features: list[str]
    features_sha256: str
    folds: int
    rows: dict[str, int]  # rows per frame
    files: dict[str, str]  # pinned file -> sha256
    frames_sha256: dict[
        str, str
    ]  # frame -> sha256 of frames/<frame>.parquet (read_frame checks it)
    windows_sha256: str  # sha256 of windows.json (read_windows checks it)
    leakage_flags: list[LeakageFlag]
    regime_flags: list[RegimeFlag]
    pinned_at: str


class Champion(_Contract):
    """``champion.json`` (§10.1): the explicit version ``score`` loads, never whatever the alias says later."""

    model_name: str
    version: int | None  # None: no champion yet, nothing is scored
    source: Literal["decide", "gate", "promote"]


class CandidateResult(_Contract):
    candidate: Candidate
    oof_pr_auc: float
    oof_brier: float
    bias_check_passed: bool
    note: str = ""


class Selection(_Contract):
    """``selection.json`` (§4.3): the candidate chosen on out-of-fold numbers, never on the test frame."""

    selected: Candidate
    candidates: list[CandidateResult] = Field(min_length=1)
    reason: str


class FeatureDtype(_Contract):
    name: str
    dtype: Literal["f64", "i64", "bool", "str"]


class PackageMeta(_Contract):
    """``meta.json`` inside every model package (§4.5). ``package_format`` is checked on load."""

    package_format: Literal[1] = 1
    candidate: Candidate
    features: list[FeatureDtype]
    categorical_levels: dict[
        str, list[str]
    ]  # levels fixed at fit time; an unseen level is treated as missing
    member_weights: dict[str, float]
    training_months: dict[Literal["eval", "refit"], list[str]]
    recipe_hash: str
    library_versions: dict[str, str]
