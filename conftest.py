"""Ensure the project root is importable in tests, regardless of how pytest is
invoked (`pytest` vs `python -m pytest`). Without this, `from api.main import app`
and `from mlops import ...` fail to collect under a bare `pytest` call in CI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
