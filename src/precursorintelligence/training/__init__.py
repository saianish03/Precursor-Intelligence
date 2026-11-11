"""Model training, calibration, tuning, and the `precursor-pipeline ml` commands (docs/ML_PIPELINE_SPEC.md)."""


class MlInputError(RuntimeError):
    """The pinned gold or the frames break a rule of spec §1.5 or §3: a data refusal (exit 2, no retry)."""


class GoldPinError(RuntimeError):
    """Gold is not current, changed during the copy, or fails a hash check (exit 1: the DAG retries)."""
