"""Evaluation metrics wrappers (SacreBLEU, chrF++, WER/CER)."""

from __future__ import annotations


def sacrebleu(hypotheses: list[str], references: list[str]) -> float:
    raise NotImplementedError("Wire sacrebleu in Notebook 06.")


def chrfpp(hypotheses: list[str], references: list[str]) -> float:
    raise NotImplementedError("Wire sacrebleu.CHRF in Notebook 06.")


def wer(hypotheses: list[str], references: list[str]) -> float:
    raise NotImplementedError("Wire jiwer/editdistance in Notebook 02/06.")
