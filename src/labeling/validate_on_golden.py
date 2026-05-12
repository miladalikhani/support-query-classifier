"""Validate the teacher prompt on a stratified subset of the golden test set.

Produces the **teacher accuracy ceiling** for the engagement (design §4.2).
Run this BEFORE committing to a full batch labeling job — if the ceiling is
below ~85%, iterate on the taxonomy descriptions ([P3][T1]) or the prompt
([P3][T3]) and re-run. Each run costs ~$0.05 for N=300.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from src.evaluation.golden import GoldenSet, load_golden
from src.labeling.client import (
    USD_PER_M_INPUT_TOKENS,
    USD_PER_M_OUTPUT_TOKENS,
    TeacherClient,
    from_env,
)
from src.labeling.labeler import LabeledMessage, label_message
from src.labeling.prompt import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    INSTRUCTION_FOOTER,
    INSTRUCTION_HEADER,
    PROMPT_VERSION,
    build_response_schema,
    prompt_fingerprint,
)
from src.labeling.taxonomy import format_class_list

DEFAULT_OUTPUT_BUCKET = "datatonic-496102-sqc-dev-artifacts"
DEFAULT_OUTPUT_PREFIX = "labeling/golden_validation/v1"

# Heuristic constants for the cost estimate (no Gemini call needed).
_CHARS_PER_TOKEN = 4
_AVG_MESSAGE_TOKENS = 20
_PROMPT_OVERHEAD_TOKENS = 60
_AVG_OUTPUT_TOKENS = 15

log = structlog.get_logger()

# (row_index, LabeledMessage, gold_label_id)
Result = tuple[int, LabeledMessage, int]


def stratified_subsample(
    df: pd.DataFrame,
    n: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic N-row subsample preserving class proportions.

    Stratified splitting requires at least one example per class, so if
    `n < num_classes` we fall back to a plain random sample.
    """
    from sklearn.model_selection import train_test_split

    if n >= len(df):
        return df.reset_index(drop=True)
    n_classes = df["label"].nunique()
    if n < n_classes:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    _, subset = train_test_split(
        df,
        test_size=n,
        random_state=seed,
        stratify=df["label"],
    )
    return subset.reset_index(drop=True)


def estimate_cost_usd(n_examples: int, taxonomy_block: str) -> tuple[float, int, int]:
    """Heuristic cost estimate based on taxonomy size. Returns (usd, avg_in, avg_out)."""
    avg_in = len(taxonomy_block) // _CHARS_PER_TOKEN + _AVG_MESSAGE_TOKENS + _PROMPT_OVERHEAD_TOKENS
    avg_out = _AVG_OUTPUT_TOKENS
    per_call = (avg_in * USD_PER_M_INPUT_TOKENS + avg_out * USD_PER_M_OUTPUT_TOKENS) / 1_000_000
    return per_call * n_examples, avg_in, avg_out


def _row_cost(labeled: LabeledMessage) -> float:
    return (
        labeled.input_tokens * USD_PER_M_INPUT_TOKENS
        + labeled.output_tokens * USD_PER_M_OUTPUT_TOKENS
    ) / 1_000_000


def label_concurrently(
    client: TeacherClient,
    examples: Sequence[tuple[str, int]],
    taxonomy_block: str,
    label_to_id: dict[str, int],
    concurrency: int = 5,
    max_cost_usd: float = 1.0,
    progress_every: int = 25,
) -> list[Result]:
    """Label examples concurrently. Halts if running cost exceeds the cap."""
    results: list[Result] = []
    cost_so_far = 0.0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(label_message, client, msg, taxonomy_block, label_to_id): (idx, gold)
            for idx, (msg, gold) in enumerate(examples)
        }

        for completed_idx, future in enumerate(as_completed(futures)):
            idx, gold = futures[future]
            try:
                labeled = future.result()
            except Exception as e:
                log.error("label_failed", index=idx, error=str(e))
                continue
            results.append((idx, labeled, gold))
            cost_so_far += _row_cost(labeled)

            done = completed_idx + 1
            if done % progress_every == 0:
                correct = sum(1 for _, lm, g in results if lm.teacher_intent_id == g)
                acc = correct / len(results) if results else 0.0
                print(f"  labeled {done}/{len(examples)}  acc={acc:.3f}  cost=${cost_so_far:.4f}")

            if cost_so_far > max_cost_usd:
                log.error(
                    "cost_cap_exceeded",
                    cost_so_far=round(cost_so_far, 4),
                    cap=max_cost_usd,
                )
                executor.shutdown(wait=False, cancel_futures=True)
                break

    return sorted(results, key=lambda r: r[0])


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    xs = sorted(values)
    if not xs:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def pct(p: float) -> float:
        # Linear interpolation between order statistics.
        if len(xs) == 1:
            return xs[0]
        k = p * (len(xs) - 1)
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return {"p50": round(pct(0.50), 1), "p95": round(pct(0.95), 1), "p99": round(pct(0.99), 1)}


def _build_prompt_snapshot(
    valid_intents: list[str],
    taxonomy_block: str,
) -> dict[str, Any]:
    """Self-contained record of every input that determines teacher behavior.

    Written alongside report.json so a run is reconstructable from artifacts
    alone — no need to consult git or rerun the labeler.
    """
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint": prompt_fingerprint(valid_intents=valid_intents),
        "model": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "instruction_header": INSTRUCTION_HEADER,
        "instruction_footer": INSTRUCTION_FOOTER,
        "taxonomy_block": taxonomy_block,
        "response_schema": build_response_schema(valid_intents),
    }


def compute_report(
    results: Sequence[Result],
    id_to_label: dict[int, str],
    model_version: str,
    wall_time_s: float,
    run_id: str,
    prompt_version: str = PROMPT_VERSION,
    prompt_fp: str = "",
) -> dict[str, Any]:
    """Roll up per-call data into the report dict."""
    valid = [(lm, g) for _, lm, g in results if lm.teacher_intent_id != -1]
    correct = sum(1 for lm, g in valid if lm.teacher_intent_id == g)
    overall_acc = correct / len(valid) if valid else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    for _, lm, g in results:
        cls = id_to_label[g]
        bucket = per_class.setdefault(cls, {"n": 0, "correct": 0})
        bucket["n"] = int(bucket["n"]) + 1
        if lm.teacher_intent_id == g:
            bucket["correct"] = int(bucket["correct"]) + 1
    for bucket in per_class.values():
        bucket["accuracy"] = round(int(bucket["correct"]) / int(bucket["n"]), 4)

    latencies = [lm.latency_ms for _, lm, _ in results]
    cost = sum(_row_cost(lm) for _, lm, _ in results)
    errors = sum(1 for _, lm, _ in results if lm.error is not None)
    input_tokens = sum(lm.input_tokens for _, lm, _ in results)
    output_tokens = sum(lm.output_tokens for _, lm, _ in results)

    return {
        "run_id": run_id,
        "model": model_version,
        "prompt": {
            "version": prompt_version,
            "fingerprint": prompt_fp,
        },
        "subset_size": len(results),
        "ceiling_top1_accuracy": round(overall_acc, 4),
        "ceiling_note": (
            "Teacher top-1 accuracy on the locked golden subset; this is the "
            "upper bound for the distilled student."
        ),
        "wall_time_s": round(wall_time_s, 1),
        "cost_usd": round(cost, 4),
        "error_count": errors,
        "tokens": {
            "input_total": input_tokens,
            "output_total": output_tokens,
        },
        "latency_ms": _percentiles(latencies),
        "per_class": per_class,
    }


def upload_to_gcs(local_path: Path, bucket_name: str, gcs_path: str) -> None:
    """Upload a single file to GCS."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))


def _build_predictions_dataframe(
    results: Sequence[Result],
    golden: GoldenSet,
    prompt_version: str,
    prompt_fp: str,
) -> pd.DataFrame:
    rows = [
        {
            **asdict(lm),
            "gold_label_id": g,
            "gold_label_name": golden.id_to_label[g],
            "correct": lm.teacher_intent_id == g,
            "prompt_version": prompt_version,
            "prompt_fingerprint": prompt_fp,
        }
        for _, lm, g in results
    ]
    return pd.DataFrame(rows)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-bucket", default=DEFAULT_OUTPUT_BUCKET)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument(
        "--no-upload", action="store_true", help="Skip GCS upload (write locally only)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print("Loading golden set...")
    golden = load_golden()
    print(f"  {len(golden.examples)} examples, {len(golden.id_to_label)} classes")

    print(f"\nStratified subsample of {args.max_examples}...")
    subset = stratified_subsample(golden.examples, args.max_examples, seed=args.seed)
    print(f"  {len(subset)} examples after sampling")

    taxonomy = format_class_list(golden.id_to_label)
    valid_intents = sorted(golden.id_to_label.values())
    prompt_snapshot = _build_prompt_snapshot(valid_intents, taxonomy)
    prompt_fp = prompt_snapshot["prompt_fingerprint"]

    print(f"\nPrompt: version={PROMPT_VERSION}  fingerprint={prompt_fp}")

    print("\nEstimating cost...")
    est_cost, avg_in, avg_out = estimate_cost_usd(len(subset), taxonomy)
    print(f"  Estimated: ~${est_cost:.4f} (avg ~{avg_in} input + {avg_out} output tokens)")
    print(f"  Hard cap:  ${args.max_cost_usd:.2f}")

    if not args.yes:
        confirm = input(f"\nContinue with {len(subset)} examples (~${est_cost:.4f})? [y/N]: ")
        if confirm.lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    print(f"\nRun ID: {run_id}")

    os.environ["MAX_CALLS_PER_RUN"] = str(len(subset) + 50)
    TeacherClient.reset_call_count()
    client = from_env()

    examples = [(str(row["text"]), int(row["label"])) for _, row in subset.iterrows()]

    print(f"Labeling {len(examples)} examples (concurrency={args.concurrency})...")
    start = time.monotonic()
    results = label_concurrently(
        client=client,
        examples=examples,
        taxonomy_block=taxonomy,
        label_to_id=golden.label_to_id,
        concurrency=args.concurrency,
        max_cost_usd=args.max_cost_usd,
    )
    wall_time = time.monotonic() - start
    print(f"\nLabeling complete: {len(results)} successful in {wall_time:.1f}s")

    report = compute_report(
        results,
        golden.id_to_label,
        client._model,
        wall_time,
        run_id,
        prompt_version=PROMPT_VERSION,
        prompt_fp=prompt_fp,
    )

    local_dir = Path("data") / "golden_validation" / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "report.json").write_text(json.dumps(report, indent=2))
    (local_dir / "prompt_snapshot.json").write_text(json.dumps(prompt_snapshot, indent=2))
    pred_df = _build_predictions_dataframe(results, golden, PROMPT_VERSION, prompt_fp)
    pred_df.to_parquet(local_dir / "predictions.parquet", index=False)

    if not args.no_upload:
        print(f"\nUploading to gs://{args.output_bucket}/{args.output_prefix}/{run_id}/...")
        for filename in ("report.json", "predictions.parquet", "prompt_snapshot.json"):
            gcs_path = f"{args.output_prefix}/{run_id}/{filename}"
            upload_to_gcs(local_dir / filename, args.output_bucket, gcs_path)
            print(f"  ✓ gs://{args.output_bucket}/{gcs_path}")

    print("\n" + "=" * 60)
    print(f"TEACHER ACCURACY CEILING: {report['ceiling_top1_accuracy']:.1%}")
    print(f"Cost:       ${report['cost_usd']:.4f}")
    print(f"Wall time:  {report['wall_time_s']:.1f}s")
    print(f"Errors:     {report['error_count']}/{report['subset_size']}")
    p = report["latency_ms"]
    print(f"Latency:    p50={p['p50']:.0f}ms  p95={p['p95']:.0f}ms  p99={p['p99']:.0f}ms")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
