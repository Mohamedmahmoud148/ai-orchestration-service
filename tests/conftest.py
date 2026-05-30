"""
tests/conftest.py — shared pytest fixtures and environment setup.

This file is auto-loaded by pytest. Its main job:
    - Set the env vars that `app.core.config.Settings` requires at import time.
    - Without `BACKEND_BASE_URL`, importing anything from `app.` raises
      RuntimeError. We set dummies so unit tests don't need a real backend.
"""
import os

os.environ.setdefault("BACKEND_BASE_URL", "http://localhost:5000")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-used-in-unit-tests")
os.environ.setdefault("ENVIRONMENT", "development")

import pytest
