"""evaluation/baselines.py: the simple rules the model is compared with (docs/ML_PIPELINE_SPEC.md §6.2)."""

import numpy as np
import polars as pl
import pytest

from precursorintelligence.evaluation.baselines import (
    BASELINES,
    baseline_scores,
    prior_gplus_rule,
    severity_24m,
    star_rating,
)
from precursorintelligence.training import MlInputError


def rows(**cols) -> pl.DataFrame:
    return pl.DataFrame(cols)


def test_prior_gplus_rule_flags_a_last_survey_hit_or_any_recent_one():
    df = rows(
        gplus_last_std=[True, False, False, None, None],
        n_gplus_surveys_36m=[0, 2, 0, None, 1],
    )
    assert prior_gplus_rule(df).tolist() == [1.0, 1.0, 0.0, 0.0, 1.0]


def test_severity_ranks_by_points_with_missing_as_zero():
    df = rows(sev_points_24m=[12.0, None, 3.5])
    assert severity_24m(df).tolist() == [12.0, 0.0, 3.5]


def test_fewer_stars_score_higher_and_unrated_homes_get_the_mean():
    df = rows(overall_rating=[1.0, 5.0, None, 3.0])
    s = star_rating(df)
    assert s[0] > s[3] > s[1]
    assert s[2] == pytest.approx(-3.0)  # the mean of 1, 5 and 3


def test_a_frame_without_any_rating_scores_zero():
    df = rows(overall_rating=pl.Series([None, None], dtype=pl.Float64))
    assert star_rating(df).tolist() == [0.0, 0.0]


def test_baseline_scores_has_every_baseline_with_one_score_per_row():
    df = rows(
        gplus_last_std=[True, False],
        n_gplus_surveys_36m=[0, 0],
        sev_points_24m=[1.0, 2.0],
        overall_rating=[2.0, 4.0],
    )
    scores = baseline_scores(df)
    assert set(scores) == set(BASELINES) == {"prior_gplus_rule", "severity_24m", "star_rating"}
    assert all(
        isinstance(s, np.ndarray) and s.dtype == np.float64 and len(s) == 2 for s in scores.values()
    )


def test_a_missing_column_is_a_data_refusal():
    with pytest.raises(MlInputError, match="sev_points_24m"):
        severity_24m(rows(other=[1]))
