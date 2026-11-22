from pathlib import Path

import pytest
from month_fakes import Month, make_month


@pytest.fixture
def month(tmp_path: Path) -> Month:
    return make_month(tmp_path)
