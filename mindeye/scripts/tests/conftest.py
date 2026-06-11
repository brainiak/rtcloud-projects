"""Pytest path setup so tests can import the project's flat-layout modules.

Adds the `scripts/` dir (for utils_mindeye, cpd_analysis) and the `models/` dir
(so utils_mindeye's top-level `import generative_models` resolves) to sys.path.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

for _p in (SCRIPTS_DIR, MODELS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
