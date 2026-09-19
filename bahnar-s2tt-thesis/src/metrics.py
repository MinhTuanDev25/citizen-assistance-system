"""Evaluation metrics wrappers (SacreBLEU, chrF++, WER/CER)."""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence


def sacrebleu_version() -> str:
    import sacrebleu as sb

    return str(getattr(sb, "__version__", "unknown"))


def _metric_signature(metric_obj: Any) -> str:
    """Best-effort SacreBLEU signature string after corpus_* has been computed."""
    if metric_obj is None:
        return ""
    # Prefer metric-class get_signature when available (needs refs count).
    get_sig = getattr(metric_obj, "get_signature", None)
    if callable(get_sig):
        try:
            return str(get_sig())
        except Exception:
            pass
    sig = getattr(metric_obj, "signature", None)
    if sig is not None:
        return str(sig)
    return ""


def sacrebleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Corpus BLEU via sacrebleu (references as single-ref list)."""
    import sacrebleu as sb

    if len(hypotheses) != len(references):
        raise ValueError("hypotheses/references length mismatch")
    if not hypotheses:
        return 0.0
    metric = sb.corpus_bleu(list(hypotheses), [list(references)])
    return float(metric.score)


def chrfpp(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Corpus chrF++ via sacrebleu."""
    import sacrebleu as sb

    if len(hypotheses) != len(references):
        raise ValueError("hypotheses/references length mismatch")
    if not hypotheses:
        return 0.0
    metric = sb.corpus_chrf(list(hypotheses), [list(references)], word_order=2)
    return float(metric.score)


def metrics_are_finite_values(*values: Any) -> bool:
    for v in values:
        try:
            if not math.isfinite(float(v)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def mt_corpus_metrics(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> Dict[str, Any]:
    import sacrebleu as sb

    bleu_score = None
    chrf_score = None
    bleu_sig = ""
    chrf_sig = ""
    bleu_cfg: Dict[str, Any] = {}
    chrf_cfg: Dict[str, Any] = {}

    if hypotheses:
        bleu_metric = sb.BLEU()
        bleu_score = bleu_metric.corpus_score(list(hypotheses), [list(references)])
        try:
            bleu_sig = str(bleu_metric.get_signature())
        except Exception:
            bleu_sig = _metric_signature(bleu_score)
        bleu_cfg = {
            "tokenize": getattr(bleu_metric, "tokenize", None),
            "smooth_method": getattr(bleu_metric, "smooth_method", None),
            "lowercase": getattr(bleu_metric, "lc", None),
        }

        chrf_metric = sb.CHRF(word_order=2)
        chrf_score = chrf_metric.corpus_score(list(hypotheses), [list(references)])
        try:
            chrf_sig = str(chrf_metric.get_signature())
        except Exception:
            chrf_sig = _metric_signature(chrf_score)
        chrf_cfg = {
            "word_order": getattr(chrf_metric, "word_order", 2),
            "beta": getattr(chrf_metric, "beta", None),
        }

    bleu = float(bleu_score.score) if bleu_score is not None else 0.0
    chrf = float(chrf_score.score) if chrf_score is not None else 0.0
    return {
        "sacrebleu": bleu,
        "chrfpp": chrf,
        "n": len(hypotheses),
        "finite": metrics_are_finite_values(bleu, chrf),
        "sacrebleu_version": sacrebleu_version(),
        "sacrebleu_signature": bleu_sig,
        "chrf_signature": chrf_sig,
        "sacrebleu_config": bleu_cfg,
        "chrf_config": chrf_cfg,
    }


def wer(hypotheses: list[str], references: list[str]) -> float:
    raise NotImplementedError("Wire jiwer/editdistance in Notebook 02/06.")
