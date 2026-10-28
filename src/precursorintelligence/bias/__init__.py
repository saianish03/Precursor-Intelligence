"""Data-level bias slicing on gold/facility_month_dataset (Ticket 5; ``precursor-pipeline bias-report``).

Slices are facility attributes and state inspection practice, not protected classes: the checks describe
differences between kinds of facilities, never discrimination. Model-level slice metrics belong to the ML phase.
"""


class BiasInputError(ValueError):
    """The dataset cannot be analysed: a required column is missing, or no rows are left in scope (exit 2)."""
