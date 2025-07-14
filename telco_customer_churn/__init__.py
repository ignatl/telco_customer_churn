"""This module contains the main directory and data directories for the telco customer churn project."""

from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent
DATA_DIR = MAIN_DIR / "data"
