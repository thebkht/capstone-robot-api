"""Ensure project-local modules (like the httpx shim) are importable."""

import os
import sys

PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
