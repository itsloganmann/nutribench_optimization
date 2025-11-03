"""Smoke tests for NutriBench prompt optimization utilities."""

from __future__ import annotations

import importlib


def test_import_optimize() -> None:
    module = importlib.import_module("src.optimize")
    assert hasattr(module, "run")


def test_import_utils() -> None:
    module = importlib.import_module("src.utils")
    assert hasattr(module, "evaluate_prompt")
