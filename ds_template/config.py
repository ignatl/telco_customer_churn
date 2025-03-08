"""Configuration management module.

This module handles configuration settings and environment variables
for the data science project.
"""

import logging
import os
from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent

log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=log_level, format="%(asctime)s - %(module)s - %(levelname)s - %(message)s")

logger = logging.getLogger("ds_template")
