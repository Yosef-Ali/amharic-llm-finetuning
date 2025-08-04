"""Evaluation module for Amharic text generation."""

from .metrics import AmharicTextEvaluator, evaluate_generation_quality

__all__ = ["AmharicTextEvaluator", "evaluate_generation_quality"]