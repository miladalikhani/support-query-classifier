"""Package trained models into self-contained, GCS-uploadable bundles.

A bundle is a directory that downstream consumers (evaluation, serving) can
load as a single unit. Layout per model:

    distilbert bundle:
        manifest.json
        label_maps.json
        temperature.json
        config.json           model.safetensors
        tokenizer.json        tokenizer_config.json
        special_tokens_map.json   vocab.txt

    baseline bundle:
        manifest.json
        label_maps.json
        temperature.json
        encoder_name.txt
        classifier.joblib

The manifest carries enough provenance to identify how the model was
produced: git SHA + dirty flag, hashed training config, teacher-label run
IDs and prompt fingerprint the model was trained against, framework
versions, and held-out val metrics.

Baseline bundles deliberately do not ship the sentence-transformer
weights — they get re-downloaded by name at load time. This keeps the
bundle small (~MB instead of ~100MB) and avoids duplicating frozen
weights across every retraining.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import structlog

from src.training.baseline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_ENCODER,
    BaselineModel,
    _embed,
    train_baseline,
)
from src.training.calibration import (
    calibrate_logits,
    compute_ece,
    fit_temperature,
)
from src.training.data import load_training_data

log = structlog.get_logger()

DEFAULT_BUCKET = "datatonic-496102-sqc-dev-artifacts"
DEFAULT_PREFIX = "models"
SCHEMA_VERSION = "v1"
DEFAULT_OUTPUT_ROOT = Path("data") / "bundles"

# Inference-time files copied from a DistilBERT training directory; training
# scratch like checkpoint-*/ and training_args.bin is intentionally dropped.
_DISTILBERT_REQUIRED = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
# Optional tokenizer files: legacy slow-tokenizer artifacts that may or may
# not be emitted depending on transformers version. Copy them if present.
_DISTILBERT_OPTIONAL = (
    "special_tokens_map.json",
    "vocab.txt",
)


# ---------- Provenance helpers ----------


def _git_info() -> tuple[str, bool]:
    """Return (sha, dirty). Falls back to ('unknown', False) outside a repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return sha, bool(porcelain.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def _hash_config(config: dict[str, Any]) -> str:
    """16-char SHA-256 over a JSON-serialised config."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _framework_versions() -> dict[str, str]:
    import platform

    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "sklearn"):
        try:
            mod = __import__(name)
            versions[name] = mod.__version__
        except ImportError:
            pass
    return versions


def _extract_run_id(uri: str) -> str:
    """gs://.../v1/<split>/<run_id>/labels.parquet → <run_id>."""
    return Path(uri).parent.name


def _build_manifest(
    *,
    model_name: str,
    teacher_train_uri: str,
    teacher_val_uri: str,
    prompt_version: str,
    prompt_fingerprint: str,
    training_config: dict[str, Any],
    val_accuracy_vs_teacher: float,
    val_accuracy_vs_truth: float,
    ece_pre: float,
    ece_post: float,
    temperature: float,
    trained_at_utc: str,
) -> dict[str, Any]:
    git_sha, git_dirty = _git_info()
    bundled_at = datetime.now(UTC).isoformat()
    return {
        "run_id": bundled_at[:19].replace(":", "-") + "Z",
        "schema_version": SCHEMA_VERSION,
        "model_name": model_name,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "training_config_hash": _hash_config(training_config),
        "teacher_labels": {
            "train_run_id": _extract_run_id(teacher_train_uri),
            "val_run_id": _extract_run_id(teacher_val_uri),
            "train_uri": teacher_train_uri,
            "val_uri": teacher_val_uri,
            "prompt_version": prompt_version,
            "prompt_fingerprint": prompt_fingerprint,
        },
        "training_metrics": {
            "val_accuracy_vs_teacher": round(val_accuracy_vs_teacher, 4),
            "val_accuracy_vs_truth": round(val_accuracy_vs_truth, 4),
            "ece_pre_scaling": round(ece_pre, 4),
            "ece_post_scaling": round(ece_post, 4),
            "temperature": round(temperature, 4),
        },
        "trained_at_utc": trained_at_utc,
        "bundled_at_utc": bundled_at,
        "framework_versions": _framework_versions(),
    }


def _write_label_maps(bundle_dir: Path, id_to_label: dict[int, str]) -> None:
    (bundle_dir / "label_maps.json").write_text(
        json.dumps(
            {
                "id_to_label": {str(k): v for k, v in id_to_label.items()},
                "label_to_id": {v: k for k, v in id_to_label.items()},
            },
            indent=2,
        )
    )


# ---------- DistilBERT bundle ----------


def save_distilbert_bundle(
    run_dir: Path,
    bundle_dir: Path,
    *,
    id_to_label: dict[int, str],
    teacher_train_uri: str,
    teacher_val_uri: str,
    prompt_version: str,
    prompt_fingerprint: str,
    training_config: dict[str, Any],
    val_accuracy_vs_teacher: float,
    val_accuracy_vs_truth: float,
    ece_pre: float,
    ece_post: float,
    temperature: float,
    trained_at_utc: str,
) -> Path:
    """Copy inference-required files from `run_dir`, add label maps + manifest."""
    bundle_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for filename in (*_DISTILBERT_REQUIRED, "temperature.json"):
        src = run_dir / filename
        if not src.exists():
            missing.append(filename)
            continue
        shutil.copy2(src, bundle_dir / filename)
    if missing:
        raise FileNotFoundError(
            f"DistilBERT run_dir {run_dir} is missing required files: {missing}"
        )
    for filename in _DISTILBERT_OPTIONAL:
        src = run_dir / filename
        if src.exists():
            shutil.copy2(src, bundle_dir / filename)

    _write_label_maps(bundle_dir, id_to_label)
    manifest = _build_manifest(
        model_name="distilbert",
        teacher_train_uri=teacher_train_uri,
        teacher_val_uri=teacher_val_uri,
        prompt_version=prompt_version,
        prompt_fingerprint=prompt_fingerprint,
        training_config=training_config,
        val_accuracy_vs_teacher=val_accuracy_vs_teacher,
        val_accuracy_vs_truth=val_accuracy_vs_truth,
        ece_pre=ece_pre,
        ece_post=ece_post,
        temperature=temperature,
        trained_at_utc=trained_at_utc,
    )
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        "saved_distilbert_bundle",
        bundle_dir=str(bundle_dir),
        run_id=manifest["run_id"],
        git_sha=manifest["git_sha"][:8],
        git_dirty=manifest["git_dirty"],
    )
    return bundle_dir


def load_distilbert_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Reload a DistilBERT bundle ready for inference. GCS URIs not supported here —
    callers should `download_bundle` into a local dir first."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    temperature = json.loads((bundle_dir / "temperature.json").read_text())["T"]
    label_maps = json.loads((bundle_dir / "label_maps.json").read_text())
    id_to_label = {int(k): v for k, v in label_maps["id_to_label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(bundle_dir))
    return {
        "model": model,
        "tokenizer": tokenizer,
        "temperature": float(temperature),
        "id_to_label": id_to_label,
        "label_to_id": {v: k for k, v in id_to_label.items()},
        "manifest": manifest,
    }


# ---------- Baseline bundle ----------


def save_baseline_bundle(
    model: BaselineModel,
    bundle_dir: Path,
    *,
    teacher_train_uri: str,
    teacher_val_uri: str,
    prompt_version: str,
    prompt_fingerprint: str,
    training_config: dict[str, Any],
    val_accuracy_vs_teacher: float,
    val_accuracy_vs_truth: float,
    ece_pre: float,
    ece_post: float,
    temperature: float,
) -> Path:
    """Persist the sklearn classifier, encoder name, calibration scalar, manifest."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.classifier, bundle_dir / "classifier.joblib")
    (bundle_dir / "encoder_name.txt").write_text(model.encoder_name + "\n")
    (bundle_dir / "temperature.json").write_text(json.dumps({"T": temperature}, indent=2))
    _write_label_maps(bundle_dir, model.id_to_label)

    manifest = _build_manifest(
        model_name="baseline_minilm_lr",
        teacher_train_uri=teacher_train_uri,
        teacher_val_uri=teacher_val_uri,
        prompt_version=prompt_version,
        prompt_fingerprint=prompt_fingerprint,
        training_config=training_config,
        val_accuracy_vs_teacher=val_accuracy_vs_teacher,
        val_accuracy_vs_truth=val_accuracy_vs_truth,
        ece_pre=ece_pre,
        ece_post=ece_post,
        temperature=temperature,
        trained_at_utc=model.trained_at_utc,
    )
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        "saved_baseline_bundle",
        bundle_dir=str(bundle_dir),
        run_id=manifest["run_id"],
        encoder=model.encoder_name,
    )
    return bundle_dir


def load_baseline_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Reload a baseline bundle. The sentence-transformer encoder is referenced by
    name — load it separately when needed (avoids duplicating ~100MB of weights)."""
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    temperature = json.loads((bundle_dir / "temperature.json").read_text())["T"]
    label_maps = json.loads((bundle_dir / "label_maps.json").read_text())
    id_to_label = {int(k): v for k, v in label_maps["id_to_label"].items()}
    encoder_name = (bundle_dir / "encoder_name.txt").read_text().strip()
    classifier = joblib.load(bundle_dir / "classifier.joblib")
    return {
        "classifier": classifier,
        "encoder_name": encoder_name,
        "temperature": float(temperature),
        "id_to_label": id_to_label,
        "label_to_id": {v: k for k, v in id_to_label.items()},
        "manifest": manifest,
    }


# ---------- Upload ----------


def upload_bundle(local_dir: Path, bucket: str, gcs_prefix: str) -> str:
    """Upload every file in `local_dir` (non-recursive) under gs://bucket/gcs_prefix/."""
    from google.cloud import storage

    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    for path in sorted(local_dir.iterdir()):
        if path.is_file():
            blob = bucket_obj.blob(f"{gcs_prefix}/{path.name}")
            blob.upload_from_filename(str(path))
    uri = f"gs://{bucket}/{gcs_prefix}/"
    log.info("uploaded_bundle", local=str(local_dir), gcs=uri)
    return uri


# ---------- CLI ----------


def _orchestrate_baseline(
    train_uri: str,
    val_uri: str,
    encoder: str,
    cache_dir: Path,
    bundle_dir: Path,
) -> dict[str, Any]:
    """Train the baseline, calibrate it, return everything needed for the manifest."""
    import numpy as np

    data = load_training_data(train_uri, val_uri)
    model = train_baseline(data, encoder_name=encoder, cache_dir=cache_dir)
    val_embeddings = _embed(data.val_texts, encoder, cache_dir)
    logits = model.classifier.decision_function(val_embeddings)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)

    val_true = np.asarray(data.val_true_labels)
    val_teacher = np.asarray(data.val_teacher_labels)
    preds = logits.argmax(axis=1)
    # decision_function returns columns ordered by classifier.classes_; convert.
    pred_class_ids = model.classifier.classes_[preds]
    vs_truth = float((pred_class_ids == val_true).mean())
    vs_teacher = float((pred_class_ids == val_teacher).mean())

    ece_pre = compute_ece(calibrate_logits(logits, 1.0), val_true)
    temperature = fit_temperature(logits, val_true)
    ece_post = compute_ece(calibrate_logits(logits, temperature), val_true)

    save_baseline_bundle(
        model,
        bundle_dir,
        teacher_train_uri=train_uri,
        teacher_val_uri=val_uri,
        prompt_version=data.prompt_version,
        prompt_fingerprint=data.prompt_fingerprint,
        training_config={
            "encoder_name": encoder,
            "lr_solver": "lbfgs",
            "lr_class_weight": "balanced",
            "lr_max_iter": 1000,
            "lr_random_state": 42,
        },
        val_accuracy_vs_teacher=vs_teacher,
        val_accuracy_vs_truth=vs_truth,
        ece_pre=ece_pre,
        ece_post=ece_post,
        temperature=temperature,
    )
    return {
        "val_accuracy_vs_teacher": vs_teacher,
        "val_accuracy_vs_truth": vs_truth,
        "ece_pre": ece_pre,
        "ece_post": ece_post,
        "temperature": temperature,
    }


def _orchestrate_distilbert(
    run_dir: Path,
    train_uri: str,
    val_uri: str,
    bundle_dir: Path,
) -> dict[str, Any]:
    """Package an already-trained DistilBERT run directory."""
    from src.training.calibration import calibrate_distilbert

    data = load_training_data(train_uri, val_uri)
    # Re-run calibration so the bundle always carries fresh, consistent metrics.
    result = calibrate_distilbert(run_dir, val_uri)
    save_distilbert_bundle(
        run_dir,
        bundle_dir,
        id_to_label=data.id_to_label,
        teacher_train_uri=train_uri,
        teacher_val_uri=val_uri,
        prompt_version=data.prompt_version,
        prompt_fingerprint=data.prompt_fingerprint,
        training_config={"source_run_dir": str(run_dir)},
        val_accuracy_vs_teacher=result.val_accuracy_vs_teacher,
        val_accuracy_vs_truth=result.val_accuracy_vs_truth,
        ece_pre=result.ece_pre,
        ece_post=result.ece_post,
        temperature=result.temperature,
        trained_at_utc=result.fitted_at_utc,
    )
    return {
        "val_accuracy_vs_teacher": result.val_accuracy_vs_teacher,
        "val_accuracy_vs_truth": result.val_accuracy_vs_truth,
        "ece_pre": result.ece_pre,
        "ece_post": result.ece_post,
        "temperature": result.temperature,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    db = sub.add_parser("distilbert", help="Package an existing DistilBERT run directory.")
    db.add_argument("--run-dir", required=True, type=Path)
    db.add_argument("--train-uri", required=True)
    db.add_argument("--val-uri", required=True)
    db.add_argument("--bundle-dir", type=Path, default=None)
    db.add_argument("--upload", action="store_true")
    db.add_argument("--bucket", default=DEFAULT_BUCKET)
    db.add_argument("--prefix", default=DEFAULT_PREFIX)

    bl = sub.add_parser("baseline", help="Train + calibrate + persist the baseline.")
    bl.add_argument("--train-uri", required=True)
    bl.add_argument("--val-uri", required=True)
    bl.add_argument("--encoder", default=DEFAULT_ENCODER)
    bl.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    bl.add_argument("--bundle-dir", type=Path, default=None)
    bl.add_argument("--upload", action="store_true")
    bl.add_argument("--bucket", default=DEFAULT_BUCKET)
    bl.add_argument("--prefix", default=DEFAULT_PREFIX)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")

    if args.command == "distilbert":
        bundle_dir = args.bundle_dir or (DEFAULT_OUTPUT_ROOT / "distilbert" / run_id)
        metrics = _orchestrate_distilbert(args.run_dir, args.train_uri, args.val_uri, bundle_dir)
        model_name = "distilbert"
    elif args.command == "baseline":
        bundle_dir = args.bundle_dir or (DEFAULT_OUTPUT_ROOT / "baseline_minilm_lr" / run_id)
        metrics = _orchestrate_baseline(
            args.train_uri, args.val_uri, args.encoder, args.cache_dir, bundle_dir
        )
        model_name = "baseline_minilm_lr"
    else:
        raise AssertionError(f"unhandled subcommand {args.command!r}")

    uploaded_uri = None
    if args.upload:
        uploaded_uri = upload_bundle(
            bundle_dir, args.bucket, f"{args.prefix}/{model_name}/{SCHEMA_VERSION}/{run_id}"
        )

    print()
    print("=" * 60)
    print(f"Model:                      {model_name}")
    print(f"Local bundle:               {bundle_dir}")
    if uploaded_uri:
        print(f"GCS:                        {uploaded_uri}")
    print(f"Val accuracy vs teacher:    {metrics['val_accuracy_vs_teacher']:.4f}")
    print(f"Val accuracy vs truth:      {metrics['val_accuracy_vs_truth']:.4f}")
    print(f"Temperature T:              {metrics['temperature']:.4f}")
    print(f"ECE pre / post:             {metrics['ece_pre']:.4f} / {metrics['ece_post']:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
