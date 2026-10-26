"""Splits (spec §9.5): no train or calib row's label window (t, t + H] reaches the test months.

Label windows overlap, so a training row at t sees outcomes up to t + H. The purge gap of H months is what keeps
those outcomes out of test; this checks it for every label, for many V and for other split settings.
"""

import polars as pl
import pytest

from precursorintelligence.common.config import SplitsConfig, load_labels_config
from precursorintelligence.gold.splits import add_splits, shift, split_windows

LABELS = load_labels_config()
VS = [shift("2025-01", i) for i in range(36)]  # V from 2025-01 to 2027-12
MONTHS = [shift("2021-01", i) for i in range(84)]  # every t from 2021-01 to 2027-12


@pytest.mark.parametrize(("test_months", "calib_months"), [(3, 3), (6, 3), (1, 0), (12, 6)])
@pytest.mark.parametrize("label", list(LABELS.labels))
def test_no_training_label_window_reaches_the_test_months(label, test_months, calib_months):
    cfg = SplitsConfig(
        spine_start="2021-01",
        test_months=test_months,
        calib_months=calib_months,
        required_labels=[label],
    )
    horizon = LABELS.labels[label].horizon_months
    rows = pl.DataFrame({"t": MONTHS})
    for v in VS:
        out = add_splits(rows, v, LABELS, cfg)
        test_from = split_windows(v, horizon, LABELS.buffer(label), cfg)["test"][0]
        fitted = out.filter(pl.col(f"split_{label}").is_in(["train", "calib"]))["t"].to_list()
        assert fitted, f"V={v}: nothing to fit"
        assert all(shift(t, horizon) < test_from for t in fitted), f"V={v}: a window reaches test"
        # and nothing at or after the test months is fitted
        assert max(fitted) < test_from
