"""Uniform inference interface over the three models the harness compares.

The evaluation harness needs to ask "what does model X predict on these
texts, and how confident is it?" without caring whether X is a remote
LLM, a fine-tuned transformer, or a sklearn classifier on frozen
embeddings. Each adapter wraps one of those into a `predict(texts)` call
returning a `PredictionBatch` with calibrated probabilities, top-k
shortcuts, and per-example latency.

Calibration: every adapter applies its bundle's saved temperature to the
model's logits before softmax, so consumers always see calibrated
probabilities. The teacher's output is already a hard label (Gemini's
JSON-schema response carries no per-class scores), so its probability
vector is a one-hot at the chosen intent.

Latency: per-example timings are measured inside the adapter so the
recorded numbers reflect tokenization + forward pass + softmax — the
work the production serving path will actually pay for. With
`batch_size=1`, latency is exact per call; with larger batches, batch
time is divided evenly across the examples in the batch.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import structlog

NUM_CLASSES = 77
DEFAULT_TOP_K = 5

log = structlog.get_logger()


@dataclass(frozen=True)
class PredictionBatch:
    """Identically-shaped output across all model adapters.

    `probs` is (n, NUM_CLASSES) over the full Banking77 label space;
    `top_k_indices` / `top_k_probs` are (n, k) sorted descending. Token
    counts are populated by the teacher adapter only and are used by the
    cost module to bill Gemini calls — local models leave them None.
    """

    probs: np.ndarray
    top_k_indices: np.ndarray
    top_k_probs: np.ndarray
    per_example_latency_ms: np.ndarray
    total_wall_time_s: float
    prompt_tokens: np.ndarray | None = None
    completion_tokens: np.ndarray | None = None


class ModelAdapter(Protocol):
    name: str

    def predict(self, texts: list[str]) -> PredictionBatch: ...


def _softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically-stable softmax(logits / T) over the last axis."""
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


def _top_k(probs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and probabilities of the top-k classes per row, descending."""
    k = min(k, probs.shape[1])
    # argpartition is O(n) and gives the unsorted top-k partition; we then
    # sort just those k entries per row for a stable descending order.
    part = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(probs.shape[0])[:, None]
    top_probs_unsorted = probs[row_idx, part]
    order = np.argsort(-top_probs_unsorted, axis=1)
    top_indices = np.take_along_axis(part, order, axis=1)
    top_probs = np.take_along_axis(top_probs_unsorted, order, axis=1)
    return top_indices.astype(np.int64), top_probs


def _probs_to_batch(
    probs: np.ndarray,
    per_example_latency_ms: np.ndarray,
    total_wall_time_s: float,
    top_k: int,
    prompt_tokens: np.ndarray | None = None,
    completion_tokens: np.ndarray | None = None,
) -> PredictionBatch:
    top_indices, top_probs = _top_k(probs, top_k)
    return PredictionBatch(
        probs=probs,
        top_k_indices=top_indices,
        top_k_probs=top_probs,
        per_example_latency_ms=per_example_latency_ms,
        total_wall_time_s=total_wall_time_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ---------- DistilBERT ----------


class DistilbertAdapter:
    """Loads a DistilBERT bundle and runs CPU inference with the bundle's T."""

    name = "distilbert"

    def __init__(
        self,
        bundle_dir: Path,
        *,
        batch_size: int = 1,
        max_length: int = 128,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        from src.training.persistence import load_distilbert_bundle

        loaded = load_distilbert_bundle(bundle_dir)
        self._model = loaded["model"]
        self._tokenizer = loaded["tokenizer"]
        self._temperature = float(loaded["temperature"])
        self._id_to_label: dict[int, str] = loaded["id_to_label"]
        self._batch_size = batch_size
        self._max_length = max_length
        self._top_k = top_k
        self._model.eval()

    def predict(self, texts: list[str]) -> PredictionBatch:
        import torch

        n = len(texts)
        probs = np.zeros((n, NUM_CLASSES), dtype=np.float64)
        latencies = np.zeros(n, dtype=np.float64)
        wall_start = time.monotonic()

        for start in range(0, n, self._batch_size):
            chunk = texts[start : start + self._batch_size]
            batch_start = time.monotonic()
            inputs = self._tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits.detach().cpu().numpy()
            elapsed_ms = (time.monotonic() - batch_start) * 1000.0
            probs[start : start + len(chunk)] = _softmax_with_temperature(
                logits, self._temperature
            )
            # For batch_size > 1, attribute equal time per example — accurate
            # for total throughput, not for per-call P95. Use batch_size=1
            # when measuring serving latency.
            latencies[start : start + len(chunk)] = elapsed_ms / len(chunk)

        return _probs_to_batch(
            probs=probs,
            per_example_latency_ms=latencies,
            total_wall_time_s=time.monotonic() - wall_start,
            top_k=self._top_k,
        )


# ---------- Baseline (frozen sentence-transformer + LR) ----------


class BaselineAdapter:
    """Sentence-transformer encoder + sklearn LR, with bundle T applied."""

    name = "baseline_minilm_lr"

    def __init__(
        self,
        bundle_dir: Path,
        *,
        batch_size: int = 1,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        from src.training.persistence import load_baseline_bundle

        loaded = load_baseline_bundle(bundle_dir)
        self._classifier = loaded["classifier"]
        self._encoder_name: str = loaded["encoder_name"]
        self._temperature = float(loaded["temperature"])
        self._id_to_label: dict[int, str] = loaded["id_to_label"]
        self._batch_size = batch_size
        self._top_k = top_k
        self._encoder = SentenceTransformer(self._encoder_name)
        # decision_function returns columns in classifier.classes_ order;
        # we project them onto a full 77-class matrix at predict time so the
        # PredictionBatch shape is model-agnostic.
        self._class_columns = np.asarray(self._classifier.classes_, dtype=np.int64)

    def predict(self, texts: list[str]) -> PredictionBatch:
        n = len(texts)
        probs = np.zeros((n, NUM_CLASSES), dtype=np.float64)
        latencies = np.zeros(n, dtype=np.float64)
        wall_start = time.monotonic()

        for start in range(0, n, self._batch_size):
            chunk = texts[start : start + self._batch_size]
            batch_start = time.monotonic()
            embeddings = self._encoder.encode(
                chunk,
                batch_size=len(chunk),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            partial_logits = self._classifier.decision_function(embeddings)
            if partial_logits.ndim == 1:
                partial_logits = partial_logits.reshape(-1, 1)
            full_logits = np.full(
                (len(chunk), NUM_CLASSES), fill_value=-np.inf, dtype=np.float64
            )
            full_logits[:, self._class_columns] = partial_logits
            elapsed_ms = (time.monotonic() - batch_start) * 1000.0
            probs[start : start + len(chunk)] = _softmax_with_temperature(
                full_logits, self._temperature
            )
            latencies[start : start + len(chunk)] = elapsed_ms / len(chunk)

        return _probs_to_batch(
            probs=probs,
            per_example_latency_ms=latencies,
            total_wall_time_s=time.monotonic() - wall_start,
            top_k=self._top_k,
        )


# ---------- Teacher (Gemini) ----------


class TeacherAdapter:
    """Wraps the labeling client so its outputs slot into the eval harness.

    Gemini returns a single intent string from the closed taxonomy; there
    is no per-class score. The adapter encodes that as a one-hot vector at
    the chosen intent (uniform over all classes when the teacher returns
    an unknown intent, so rows still sum to 1.0). Prompt + completion token
    counts are surfaced on the batch so the cost module can bill calls.
    """

    name = "teacher"

    def __init__(
        self,
        client: Any,
        label_to_id: dict[str, int],
        taxonomy_block: str,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._client = client
        self._label_to_id = label_to_id
        self._taxonomy_block = taxonomy_block
        self._top_k = top_k

    def predict(self, texts: list[str]) -> PredictionBatch:
        from src.labeling.labeler import label_message

        n = len(texts)
        probs = np.full((n, NUM_CLASSES), fill_value=1.0 / NUM_CLASSES, dtype=np.float64)
        latencies = np.zeros(n, dtype=np.float64)
        prompt_tokens = np.zeros(n, dtype=np.int64)
        completion_tokens = np.zeros(n, dtype=np.int64)
        wall_start = time.monotonic()

        for i, text in enumerate(texts):
            labeled = label_message(
                self._client, text, self._taxonomy_block, self._label_to_id
            )
            if labeled.teacher_intent_id >= 0:
                probs[i] = 0.0
                probs[i, labeled.teacher_intent_id] = 1.0
            latencies[i] = labeled.latency_ms
            prompt_tokens[i] = labeled.input_tokens
            completion_tokens[i] = labeled.output_tokens

        return _probs_to_batch(
            probs=probs,
            per_example_latency_ms=latencies,
            total_wall_time_s=time.monotonic() - wall_start,
            top_k=self._top_k,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
