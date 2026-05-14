"""End-to-end evaluation runner.

Loads the locked golden test set and the val split, runs the supplied
adapters, fits per-class thresholds, computes metrics, and writes a
machine-readable `results.json` plus a human-readable `report.md` into
the run directory.

The runner is pure orchestration — every numerical decision lives in
the metrics/cost/threshold modules. That separation lets tests run
end-to-end with mock adapters on tiny fixtures.

Latency-source discipline: the headline cost and latency numbers
should come from a Cloud Run load test
(`serving_latency.load_cloud_run_latencies`). When that file is not
provided, the runner falls back to the adapter's in-process latency
and the report clearly labels it "laptop_dev" so no one mistakes it
for a production claim.
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from src.evaluation.adapters import (
    BaselineAdapter,
    DistilbertAdapter,
    ModelAdapter,
    TeacherAdapter,
)
from src.evaluation.cost import (
    PRICING_DATE,
    format_cost_breakdown,
    student_cost_per_1k,
    teacher_cost_per_1k,
)
from src.evaluation.golden import GoldenSet, load_golden
from src.evaluation.metrics import (
    compute_ece,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
    reliability_diagram,
    top_confusion_pairs,
    top_k_accuracy,
)
from src.evaluation.serving_latency import CloudRunLatencies, load_cloud_run_latencies
from src.evaluation.thresholds import (
    ESCALATE_ALL_THRESHOLD,
    apply_thresholds,
    fit_per_class_thresholds,
    save_thresholds,
    summarize_thresholds,
)
from src.evaluation.val import ValSet, load_val

log = structlog.get_logger()

LABEL_NOISE_BAND_PP = 10  # Banking77's documented inherent label-noise band
DEFAULT_TOP_CONFUSION_N = 20
DEFAULT_TARGET_PRECISION = 0.95
DEFAULT_SERVING_VCPU = 1.0
DEFAULT_SERVING_MEMORY_GIB = 2.0
DEFAULT_OUTPUT_ROOT = Path("data") / "evaluation"


# ---------- Provenance ----------


def _git_info() -> tuple[str, bool]:
    """(sha, dirty); falls back to ('unknown', False) outside a repo."""
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


def _load_bundle_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ---------- Per-model evaluation ----------


def _latency_percentiles(latencies: np.ndarray) -> dict[str, float]:
    return {
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "latency_ms_p99": float(np.percentile(latencies, 99)),
    }


def _evaluate_classifier(
    adapter: ModelAdapter,
    val_texts: list[str],
    val_labels: np.ndarray,
    golden_texts: list[str],
    golden_labels: np.ndarray,
    id_to_label: dict[int, str],
    output_dir: Path,
    *,
    target_precision: float,
    cloud_run_latencies: CloudRunLatencies | None,
    serving_vcpu: float,
    serving_memory_gib: float,
) -> dict[str, Any]:
    """Eval for distilbert / baseline: thresholds, ECE, latency-based cost."""
    log.info("predict_val", model=adapter.name, n=len(val_texts))
    val_batch = adapter.predict(val_texts)
    log.info("predict_golden", model=adapter.name, n=len(golden_texts))
    golden_batch = adapter.predict(golden_texts)

    # Thresholds: fit on val, apply on golden.
    fit_result = fit_per_class_thresholds(
        val_batch.probs, val_labels, target_precision=target_precision
    )
    threshold_summary = summarize_thresholds(
        fit_result.thresholds, val_batch.probs, val_labels, id_to_label
    )
    save_thresholds(
        fit_result, threshold_summary, output_dir / f"{adapter.name}_thresholds.json"
    )

    golden_preds, golden_accepted = apply_thresholds(
        golden_batch.probs, fit_result.thresholds
    )
    correct = golden_preds == golden_labels
    n_accepted = int(golden_accepted.sum())
    n_correct_accepted = int((correct & golden_accepted).sum())

    # Reliability diagram on golden (the calibrated probabilities, not thresholded).
    reliability_diagram(
        golden_batch.probs,
        golden_labels,
        out_path=output_dir / f"reliability_{adapter.name}.png",
    )

    # Latency: prefer Cloud Run load-test data; otherwise use the in-process times.
    if cloud_run_latencies is not None:
        latencies = cloud_run_latencies.per_example_latency_ms
        latency_source = "cloud_run_load_test"
    else:
        latencies = golden_batch.per_example_latency_ms
        latency_source = "laptop_dev"

    metrics: dict[str, Any] = {
        "top_1_accuracy": top_k_accuracy(golden_batch.probs, golden_labels, k=1),
        "top_5_accuracy": top_k_accuracy(golden_batch.probs, golden_labels, k=5),
        "macro_f1": macro_f1(golden_batch.probs, golden_labels),
        "ece": compute_ece(golden_batch.probs, golden_labels, n_bins=15),
        **_latency_percentiles(latencies),
        "latency_source": latency_source,
        "cost_per_1k_usd": student_cost_per_1k(
            latencies, vcpu=serving_vcpu, memory_gib=serving_memory_gib
        ),
    }

    per_class = per_class_metrics(golden_batch.probs, golden_labels, id_to_label)
    cm = confusion_matrix(golden_batch.probs, golden_labels, normalize=True)
    confusion = top_confusion_pairs(cm, id_to_label, top_n=DEFAULT_TOP_CONFUSION_N)

    threshold_stats = {
        "target_precision": fit_result.target_precision,
        "n_classes_escalate_all": sum(
            1 for t in fit_result.thresholds.values() if t == ESCALATE_ALL_THRESHOLD
        ),
        "n_golden": len(golden_labels),
        "n_golden_accepted": n_accepted,
        "golden_acceptance_rate": float(golden_accepted.mean()),
        "golden_precision_among_accepted": (
            float(n_correct_accepted) / n_accepted if n_accepted > 0 else 0.0
        ),
    }

    return {
        "kind": "calibrated_classifier",
        "metrics": metrics,
        "thresholds": threshold_stats,
        "per_class": per_class.to_dict(orient="records"),
        "top_confusion_pairs": [
            {"true": t, "predicted": p, "value": v} for t, p, v in confusion
        ],
    }


def _evaluate_teacher(
    adapter: TeacherAdapter,
    golden_texts: list[str],
    golden_labels: np.ndarray,
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    """Eval for the LLM teacher: token-billed cost, no thresholds, no top-5."""
    log.info("predict_teacher", n=len(golden_texts))
    batch = adapter.predict(golden_texts)
    if batch.prompt_tokens is None or batch.completion_tokens is None:
        raise RuntimeError("Teacher batch missing token counts; cost cannot be computed.")

    metrics: dict[str, Any] = {
        "top_1_accuracy": top_k_accuracy(batch.probs, golden_labels, k=1),
        "macro_f1": macro_f1(batch.probs, golden_labels),
        "ece": compute_ece(batch.probs, golden_labels, n_bins=15),
        **_latency_percentiles(batch.per_example_latency_ms),
        "latency_source": "gemini_api_synchronous",
        "cost_per_1k_usd": teacher_cost_per_1k(
            batch.prompt_tokens, batch.completion_tokens
        ),
    }

    per_class = per_class_metrics(batch.probs, golden_labels, id_to_label)
    cm = confusion_matrix(batch.probs, golden_labels, normalize=True)
    confusion = top_confusion_pairs(cm, id_to_label, top_n=DEFAULT_TOP_CONFUSION_N)

    return {
        "kind": "llm_teacher",
        "metrics": metrics,
        "per_class": per_class.to_dict(orient="records"),
        "top_confusion_pairs": [
            {"true": t, "predicted": p, "value": v} for t, p, v in confusion
        ],
    }


# ---------- Orchestrator ----------


def evaluate(
    adapters: Sequence[ModelAdapter],
    val_set: ValSet,
    golden_set: GoldenSet,
    output_dir: Path,
    *,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    cloud_run_latencies_by_model: Mapping[str, CloudRunLatencies] | None = None,
    serving_vcpu: float = DEFAULT_SERVING_VCPU,
    serving_memory_gib: float = DEFAULT_SERVING_MEMORY_GIB,
    bundle_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Run every adapter against the golden set, write JSON + markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    val_texts = val_set.examples["text"].tolist()
    val_labels = np.asarray(val_set.examples["label"], dtype=np.int64)
    golden_texts = golden_set.examples["text"].tolist()
    golden_labels = np.asarray(golden_set.examples["label"], dtype=np.int64)
    id_to_label = golden_set.id_to_label

    models: dict[str, Any] = {}
    for adapter in adapters:
        latencies = (
            cloud_run_latencies_by_model.get(adapter.name)
            if cloud_run_latencies_by_model
            else None
        )
        if isinstance(adapter, TeacherAdapter):
            models[adapter.name] = _evaluate_teacher(
                adapter, golden_texts, golden_labels, id_to_label
            )
        else:
            models[adapter.name] = _evaluate_classifier(
                adapter,
                val_texts,
                val_labels,
                golden_texts,
                golden_labels,
                id_to_label,
                output_dir,
                target_precision=target_precision,
                cloud_run_latencies=latencies,
                serving_vcpu=serving_vcpu,
                serving_memory_gib=serving_memory_gib,
            )
        # Bundle provenance, when the caller supplied a path.
        if bundle_paths is not None and adapter.name in bundle_paths:
            manifest = _load_bundle_manifest(bundle_paths[adapter.name])
            models[adapter.name]["bundle_path"] = str(bundle_paths[adapter.name])
            models[adapter.name]["bundle_manifest"] = {
                "run_id": manifest.get("run_id"),
                "git_sha": manifest.get("git_sha"),
                "training_config_hash": manifest.get("training_config_hash"),
                "temperature": manifest.get("training_metrics", {}).get("temperature"),
            }

    git_sha, git_dirty = _git_info()
    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    cost_summary = _build_cost_summary(models)

    results: dict[str, Any] = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "pricing_date": PRICING_DATE,
        "target_precision": target_precision,
        "serving_shape": {"vcpu": serving_vcpu, "memory_gib": serving_memory_gib},
        "label_noise_band_pp": LABEL_NOISE_BAND_PP,
        "golden_n_examples": len(golden_labels),
        "val_n_examples": len(val_labels),
        "num_classes": len(id_to_label),
        "models": models,
        "cost_summary": cost_summary,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (output_dir / "report.md").write_text(render_report_md(results))
    log.info("evaluation_written", output_dir=str(output_dir), run_id=run_id)
    return results


def _build_cost_summary(models: Mapping[str, Any]) -> dict[str, Any]:
    """Roll up per-model cost-per-1k into the format_cost_breakdown shape."""
    teacher_cost = _model_cost(models.get("teacher"))
    student_cost = _model_cost(models.get("distilbert"))
    baseline_cost = _model_cost(models.get("baseline_minilm_lr"))
    if teacher_cost is None or student_cost is None:
        return {
            "pricing_date": PRICING_DATE,
            "note": "cost summary skipped: teacher or student missing from this run",
        }
    return format_cost_breakdown(
        teacher=teacher_cost, student=student_cost, baseline=baseline_cost
    )


def _model_cost(model_entry: Mapping[str, Any] | None) -> float | None:
    if model_entry is None:
        return None
    return float(model_entry["metrics"]["cost_per_1k_usd"])


# ---------- Markdown rendering ----------


def render_report_md(results: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Evaluation Report — {results['run_id']}")
    lines.append("")
    lines.append(
        f"Generated at {results['generated_at_utc']}  |  "
        f"git `{results['git_sha'][:8]}`"
        f"{' (dirty)' if results['git_dirty'] else ''}  |  "
        f"pricing as of {results['pricing_date']}"
    )
    lines.append("")
    lines.append(
        f"Golden examples: {results['golden_n_examples']}  |  "
        f"Val examples: {results['val_n_examples']}  |  "
        f"Classes: {results['num_classes']}"
    )
    lines.append("")
    lines.append(_headline_table(results["models"]))
    lines.append("")
    lines.append(_cost_section(results.get("cost_summary", {})))
    lines.append("")
    lines.append(_label_noise_section(results))
    lines.append("")
    for name, model in results["models"].items():
        lines.append(_per_model_section(name, model))
        lines.append("")
    return "\n".join(lines)


def _headline_table(models: Mapping[str, Any]) -> str:
    rows = [
        "## Headline Metrics",
        "",
        "| Model | Top-1 | Macro F1 | ECE | Latency P95 (ms) | Cost / 1k (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for name, model in models.items():
        m = model["metrics"]
        rows.append(
            "| {name} | {top1:.3f} | {f1:.3f} | {ece:.3f} | "
            "{p95:.0f} | ${cost:.6f} |".format(
                name=name,
                top1=m["top_1_accuracy"],
                f1=m["macro_f1"],
                ece=m["ece"],
                p95=m["latency_ms_p95"],
                cost=m["cost_per_1k_usd"],
            )
        )
    return "\n".join(rows)


def _cost_section(cost_summary: Mapping[str, Any]) -> str:
    if "note" in cost_summary:
        return "## Cost Reduction\n\n_{}_".format(cost_summary["note"])
    rows = ["## Cost Reduction vs Teacher", ""]
    student_x = cost_summary.get("student_cost_reduction_vs_teacher_x")
    if student_x is not None:
        rows.append(f"- Student vs teacher: **{student_x}x** cheaper")
    baseline_x = cost_summary.get("baseline_cost_reduction_vs_teacher_x")
    if baseline_x is not None:
        rows.append(f"- Baseline vs teacher: **{baseline_x}x** cheaper")
    rows.append("")
    rows.append("Brief target: cost reduction ≥10x.")
    return "\n".join(rows)


def _label_noise_section(results: Mapping[str, Any]) -> str:
    band = results["label_noise_band_pp"]
    lines = [
        "## Banking77 Label-Noise Framing",
        "",
        f"Banking77 carries roughly ±{band} pp of inherent label noise. Accuracy is",
        "reported raw above; the noise-adjusted ceiling for any model is roughly",
        f"`raw_accuracy + {band}pp`. Don't read the raw number as the achievable ceiling.",
    ]
    return "\n".join(lines)


def _per_model_section(name: str, model: Mapping[str, Any]) -> str:
    m = model["metrics"]
    lines = [f"## Model: `{name}`", ""]

    # Latency caveat
    latency_source = m.get("latency_source", "unknown")
    if latency_source == "laptop_dev":
        lines.append(
            "> **Caveat**: latency and student/baseline cost are from the eval host"
            " (developer laptop). Production Cloud Run vCPUs are 2-4x slower per"
            " core, so the headline cost is understated and latency is overstated"
            " (in your favor). Re-run with the Phase 6 load-test JSON for the"
            " production claim."
        )
        lines.append("")

    lines.append(
        f"- Top-1: {m['top_1_accuracy']:.3f}"
        + (f"  |  Top-5: {m['top_5_accuracy']:.3f}" if "top_5_accuracy" in m else "")
    )
    lines.append(f"- Macro F1: {m['macro_f1']:.3f}")
    lines.append(f"- ECE (15-bin): {m['ece']:.3f}")
    lines.append(
        f"- Latency P50/P95/P99: "
        f"{m['latency_ms_p50']:.0f} / {m['latency_ms_p95']:.0f} / "
        f"{m['latency_ms_p99']:.0f} ms ({latency_source})"
    )
    lines.append(f"- Cost per 1k: ${m['cost_per_1k_usd']:.6f}")
    lines.append("")

    if model.get("kind") == "calibrated_classifier":
        rel_path = f"reliability_{name}.png"
        lines.append(f"![Reliability diagram for {name}]({rel_path})")
        lines.append("")
        t = model["thresholds"]
        lines.append("### Thresholds (auto-route vs escalate)")
        lines.append("")
        lines.append(f"- Target precision: {t['target_precision']}")
        lines.append(
            f"- Classes set to escalate-all "
            f"(no qualifying threshold): {t['n_classes_escalate_all']}"
        )
        lines.append(
            f"- Auto-routed on golden: {t['n_golden_accepted']} / {t['n_golden']}"
            f"  ({t['golden_acceptance_rate']:.1%})"
        )
        lines.append(
            f"- Precision among auto-routed: "
            f"{t['golden_precision_among_accepted']:.3f}"
        )
        lines.append("")

    lines.append("### Worst classes (sorted by F1 asc)")
    lines.append("")
    lines.append("| Class | Support | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|")
    sorted_pc = sorted(model["per_class"], key=lambda row: row["f1"])[:15]
    for row in sorted_pc:
        lines.append(
            f"| {row['class_name']} | {int(row['support'])} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        )
    lines.append("")

    lines.append(f"### Top {DEFAULT_TOP_CONFUSION_N} confused pairs")
    lines.append("")
    lines.append("| True class | Predicted class | Rate |")
    lines.append("|---|---|---|")
    for pair in model["top_confusion_pairs"]:
        lines.append(
            f"| {pair['true']} | {pair['predicted']} | {pair['value']:.3f} |"
        )
    return "\n".join(lines)


# ---------- CLI ----------


def _build_adapters(args: argparse.Namespace) -> tuple[list[ModelAdapter], dict[str, Path]]:
    adapters: list[ModelAdapter] = []
    bundle_paths: dict[str, Path] = {}
    if args.distilbert_bundle is not None:
        distilbert = DistilbertAdapter(args.distilbert_bundle)
        adapters.append(distilbert)
        bundle_paths[distilbert.name] = args.distilbert_bundle
    if args.baseline_bundle is not None:
        baseline = BaselineAdapter(args.baseline_bundle)
        adapters.append(baseline)
        bundle_paths[baseline.name] = args.baseline_bundle
    if args.include_teacher:
        from src.labeling.client import from_env
        from src.labeling.taxonomy import format_class_list

        golden = load_golden()
        adapters.append(
            TeacherAdapter(
                client=from_env(),
                label_to_id=golden.label_to_id,
                taxonomy_block=format_class_list(golden.id_to_label),
            )
        )
    return adapters, bundle_paths


def _load_latency_overrides(args: argparse.Namespace) -> dict[str, CloudRunLatencies]:
    out: dict[str, CloudRunLatencies] = {}
    if args.distilbert_latencies is not None:
        out["distilbert"] = load_cloud_run_latencies(args.distilbert_latencies)
    if args.baseline_latencies is not None:
        out["baseline_minilm_lr"] = load_cloud_run_latencies(args.baseline_latencies)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distilbert-bundle", type=Path, default=None)
    parser.add_argument("--baseline-bundle", type=Path, default=None)
    parser.add_argument(
        "--include-teacher",
        action="store_true",
        help="Run the Gemini teacher on golden (costs ~$1 per run; off by default).",
    )
    parser.add_argument(
        "--distilbert-latencies",
        type=Path,
        default=None,
        help="Cloud Run load-test JSON for the student.",
    )
    parser.add_argument(
        "--baseline-latencies",
        type=Path,
        default=None,
        help="Cloud Run load-test JSON for the baseline.",
    )
    parser.add_argument("--target-precision", type=float, default=DEFAULT_TARGET_PRECISION)
    parser.add_argument("--serving-vcpu", type=float, default=DEFAULT_SERVING_VCPU)
    parser.add_argument(
        "--serving-memory-gib", type=float, default=DEFAULT_SERVING_MEMORY_GIB
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    adapters, bundle_paths = _build_adapters(args)
    if not adapters:
        print("Pass at least one of --distilbert-bundle / --baseline-bundle / --include-teacher.")
        return 1

    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / run_id)
    evaluate(
        adapters,
        val_set=load_val(),
        golden_set=load_golden(),
        output_dir=output_dir,
        target_precision=args.target_precision,
        cloud_run_latencies_by_model=_load_latency_overrides(args),
        serving_vcpu=args.serving_vcpu,
        serving_memory_gib=args.serving_memory_gib,
        bundle_paths=bundle_paths,
    )
    print(f"Wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
