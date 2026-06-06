"""FinBERT financial-sentiment scoring.

Wraps `ProsusAI/finbert <https://huggingface.co/ProsusAI/finbert>`_ — a BERT
model fine-tuned on financial text — to score news headlines/summaries in
``[-1, 1]`` (positive = bullish). The score for a piece of text is
``P(positive) - P(negative)`` from the model's softmax over its
positive/negative/neutral labels, so a confidently-neutral article lands near
0.0 just like a mixed one.

The model and tokenizer are heavy (torch + a few hundred MB of weights) and are
loaded lazily on first use, then cached, so importing this module is cheap and
code paths that never touch sentiment never pull in torch/transformers. When
those dependencies aren't installed (e.g. the 32-bit ARM Raspberry Pi target,
where torch has no wheels) loading raises :class:`FinBertUnavailable`; callers
are expected to catch it and fall back to VADER.
"""

from __future__ import annotations

import threading
from typing import Sequence

from config.logging_setup import get_logger

logger = get_logger(__name__)

# HuggingFace model id; fine-tuned on the Financial PhraseBank.
DEFAULT_MODEL_NAME = "ProsusAI/finbert"

# FinBERT was trained on single sentences; 512 is BERT's hard token ceiling and
# more than enough for a headline + a sentence or two of summary.
_MAX_TOKENS = 512


class FinBertUnavailable(RuntimeError):
    """Raised when the FinBERT model or its dependencies can't be loaded.

    Callers should treat this as "fall back to VADER sentiment", not as a
    fatal error — the bot is expected to run on hosts (e.g. the Pi) where torch
    isn't installable.
    """


class FinBertScorer:
    """Lazily-loaded FinBERT scorer returning sentiment in ``[-1, 1]``.

    Thread-safe: model load is guarded by a lock and inference runs under
    ``torch.no_grad``. A single shared instance per process is intended (see
    :func:`get_default_scorer`); load it once and reuse it.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, batch_size: int = 16) -> None:
        self._model_name = model_name
        self._batch_size = max(1, batch_size)
        self._lock = threading.Lock()
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._pos_idx = 0
        self._neg_idx = 1
        self._load_failed = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Load tokenizer + model on first use; cache the result.

        Raises :class:`FinBertUnavailable` if the dependencies are missing or the
        weights can't be fetched. Once a load fails we don't retry — subsequent
        calls raise immediately so callers stay on their fallback without
        re-probing (and re-downloading) every cycle.
        """
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if self._load_failed:
                raise FinBertUnavailable(f"FinBERT ({self._model_name}) previously failed to load")
            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as exc:
                self._load_failed = True
                raise FinBertUnavailable(
                    "transformers/torch not installed; "
                    "`pip install transformers torch` to enable FinBERT"
                ) from exc
            try:
                tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
                model.eval()
            except Exception as exc:  # noqa: BLE001 - any load failure -> fall back
                self._load_failed = True
                raise FinBertUnavailable(
                    f"failed to load FinBERT ({self._model_name}): {exc}"
                ) from exc

            self._pos_idx, self._neg_idx = self._resolve_label_indices(model.config)
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            logger.info("FinBERT loaded (%s, batch_size=%d)", self._model_name, self._batch_size)

    @staticmethod
    def _resolve_label_indices(config) -> tuple[int, int]:
        """Find the positive/negative logit indices from the model's label map.

        FinBERT ships ``{0: positive, 1: negative, 2: neutral}``, but we read it
        off the config rather than hard-coding in case a variant reorders them.
        """
        id2label = getattr(config, "id2label", None) or {0: "positive", 1: "negative", 2: "neutral"}
        pos_idx = neg_idx = None
        for idx, label in id2label.items():
            name = str(label).lower()
            if name.startswith("pos"):
                pos_idx = int(idx)
            elif name.startswith("neg"):
                neg_idx = int(idx)
        if pos_idx is None or neg_idx is None:
            raise FinBertUnavailable(f"unexpected FinBERT label map: {id2label}")
        return pos_idx, neg_idx

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score_texts(self, texts: Sequence[str | None]) -> list[float]:
        """Score each text in ``[-1, 1]`` (positive = bullish).

        Empty/blank texts score a neutral 0.0 without hitting the model. Raises
        :class:`FinBertUnavailable` if the model can't be loaded.
        """
        items = [(t or "").strip() for t in texts]
        scores = [0.0] * len(items)
        to_score = [i for i, t in enumerate(items) if t]
        if not to_score:
            return scores

        self._ensure_loaded()
        torch = self._torch
        with torch.no_grad():
            for start in range(0, len(to_score), self._batch_size):
                chunk = to_score[start : start + self._batch_size]
                batch = [items[i] for i in chunk]
                encoded = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=_MAX_TOKENS,
                )
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)
                for row, i in enumerate(chunk):
                    p = probs[row]
                    scores[i] = float(p[self._pos_idx] - p[self._neg_idx])
        return scores

    def score_text(self, text: str | None) -> float:
        """Score a single piece of text in ``[-1, 1]``."""
        return self.score_texts([text])[0]


_default_scorer: FinBertScorer | None = None
_default_lock = threading.Lock()


def get_default_scorer() -> FinBertScorer:
    """Return the process-wide shared :class:`FinBertScorer` (created on demand)."""
    global _default_scorer
    if _default_scorer is None:
        with _default_lock:
            if _default_scorer is None:
                _default_scorer = FinBertScorer()
    return _default_scorer
