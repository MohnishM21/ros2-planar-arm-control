from pathlib import Path

from ament_flake8.main import main_with_errors
import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    config = Path(__file__).resolve().parents[1] / "flake8.ini"
    rc, errors = main_with_errors(argv=["--config", str(config)])
    assert rc == 0, f"Found {len(errors)} code style errors / warnings:\n" + "\n".join(errors)
