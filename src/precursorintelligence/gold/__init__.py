"""Gold node (docs/DATA_PIPELINE_SPEC.md §7-§11): silver tables -> spine, labels, features, dataset."""


class GoldCheckError(RuntimeError):
    """A gold build broke one of its assertions (row counts, keys, coverage); nothing is published."""
