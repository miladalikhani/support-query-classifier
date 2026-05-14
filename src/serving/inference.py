"""Single-request inference path for the Cloud Run serving layer.

`InferenceService` owns the end-to-end "message in → labelled response out"
flow: PII redaction, DistilBERT forward pass, calibrated softmax, top-k
selection, response shaping. The HTTP layer in `app.py` calls one method
on this class — keeping the FastAPI handler thin and the inference path
unit-testable without spinning up a server.

The DistilBERT model loading is delegated to
`src.evaluation.adapters.DistilbertAdapter`, which already handles the
bundle (model, tokenizer, calibrated temperature) and applies
`softmax(logits / T)`. The serving layer reuses that adapter rather
than duplicating its loading code.

Bundle source:
  - Local path → used directly.
  - `gs://` URI → all files downloaded into a temporary directory at
    startup, then the local directory is handed to the adapter. The
    bundle is not baked into the Docker image so Artifact Registry
    stays under the 0.5 GB always-free tier.
"""

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.adapters import DistilbertAdapter
from src.pii.interface import Redactor


@dataclass(frozen=True)
class PredictionResponse:
    """Service-level response from `InferenceService.predict`."""

    predicted_intent: str
    confidence: float
    top_k_intents: list[str]
    top_k_confidences: list[float]
    redacted_message: str
    latency_ms: float
    model_version: str


class InferenceService:
    """Holds the loaded model and serves single-message predictions."""

    def __init__(
        self,
        bundle_uri: str | Path,
        redactor: Redactor,
        model_version: str,
        *,
        top_k: int = 5,
    ) -> None:
        local_dir = _materialize_bundle(bundle_uri)
        self._adapter = DistilbertAdapter(local_dir, top_k=top_k)
        self._id_to_label = _load_id_to_label(local_dir)
        self._redactor = redactor
        self._model_version = model_version
        self._top_k = top_k

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, message: str) -> PredictionResponse:
        start = time.monotonic()
        redaction = self._redactor.redact(message)
        batch = self._adapter.predict([redaction.redacted_text])
        elapsed_ms = (time.monotonic() - start) * 1000.0

        top_indices = batch.top_k_indices[0]
        top_probs = batch.top_k_probs[0]
        top_intents = [self._id_to_label[int(i)] for i in top_indices]
        top_confidences = [float(p) for p in top_probs]

        return PredictionResponse(
            predicted_intent=top_intents[0],
            confidence=top_confidences[0],
            top_k_intents=top_intents,
            top_k_confidences=top_confidences,
            redacted_message=redaction.redacted_text,
            latency_ms=elapsed_ms,
            model_version=self._model_version,
        )


def _materialize_bundle(bundle_uri: str | Path) -> Path:
    """Return a local directory containing the bundle files."""
    uri = str(bundle_uri)
    if uri.startswith("gs://"):
        return _download_from_gcs(uri)
    local = Path(uri)
    if not local.exists():
        raise FileNotFoundError(f"Bundle path does not exist: {local}")
    return local


def _download_from_gcs(gs_uri: str) -> Path:
    """Download every object under a gs://bucket/prefix to a temp directory."""
    from google.cloud import storage

    bucket_name, prefix = _parse_gs_uri(gs_uri)
    dest = Path(tempfile.mkdtemp(prefix="bundle_"))
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No objects found under {gs_uri}")
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        local_path = dest / Path(blob.name).name
        blob.download_to_filename(str(local_path))
    return dest


def _parse_gs_uri(gs_uri: str) -> tuple[str, str]:
    without_scheme = gs_uri[len("gs://"):]
    if "/" not in without_scheme:
        return without_scheme, ""
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix


def _load_id_to_label(bundle_dir: Path) -> dict[int, str]:
    """Read the bundle's label-map sidecar so the response carries class names."""
    label_maps = json.loads((bundle_dir / "label_maps.json").read_text())
    return {int(k): v for k, v in label_maps["id_to_label"].items()}
