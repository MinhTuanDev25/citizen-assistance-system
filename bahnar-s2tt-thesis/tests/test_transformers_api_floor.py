"""Dependency floor and TrainingArguments API signature checks."""
from __future__ import annotations

import inspect

import pytest


class TestTransformersApiFloor:
    def test_training_arguments_accepts_eval_strategy_and_processing_class(self):
        from transformers import TrainingArguments

        sig = inspect.signature(TrainingArguments.__init__)
        params = set(sig.parameters)
        assert "eval_strategy" in params or "evaluation_strategy" in params
        # processing_class is a Trainer kwarg, not always TrainingArguments
        from transformers import Trainer

        trainer_params = set(inspect.signature(Trainer.__init__).parameters)
        assert "processing_class" in trainer_params or "tokenizer" in trainer_params

    def test_installed_transformers_meets_floor(self):
        import transformers
        from packaging.version import Version

        assert Version(transformers.__version__.split("+")[0]) >= Version("4.49.0")
