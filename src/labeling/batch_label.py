"""Production-grade batch labeling of a Banking77 partition with the teacher.

Differs from `validate_on_golden.py` in:
  - Runs at full split scale (up to ~9k rows in one job)
  - Per-row JSONL checkpoint → resumable across crashes / network blips
  - `--split {val,golden,train}` selects which partition to label
  - Per-split GCS namespace at `…/teacher_labels/v1/<split>/<run_id>/`

Recommended run order: val (~1k) → golden (3k) → train (~9k). Bugs surface on
the cheapest run before we commit to the longest one.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from src.data.banking77 import load_banking77
from src.evaluation.golden import load_golden
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

VALID_SPLITS = ("val", "golden", "train")

DEFAULT_OUTPUT_BUCKET = "datatonic-496102-sqc-dev-artifacts"
DEFAULT_OUTPUT_PREFIX = "labeling/teacher_labels/v1"

# Per-split cost caps (USD). Roughly 1.5-2x the live estimate at v3 prompt size + 2.5 Flash pricing.
DEFAULT_COST_CAPS = {
    "val": 1.50,
    "golden": 4.00,
    "train": 10.00,
}

_CHECKPOINT_DIR = Path("data") / "checkpoints"

# Heuristic constants for the pre-flight cost estimate.
_CHARS_PER_TOKEN = 4
_AVG_MESSAGE_TOKENS = 20
_PROMPT_OVERHEAD_TOKENS = 60
_AVG_OUTPUT_TOKENS = 15

log = structlog.get_logger()

# (row_index, LabeledMessage, gold_label_id)
Result = tuple[int, LabeledMessage, int]


# ---------- Split loaders ----------


def load_split(split: str) -> pd.DataFrame:
    """Load the DataFrame for the named partition with `text` and `label` columns."""
    if split == "val":
        return load_banking77().val
    if split == "train":
        return load_banking77().train
    if split == "golden":
        return load_golden().examples
    raise ValueError(f"Unknown split: {split!r}. Must be one of {VALID_SPLITS}.")


def load_label_maps() -> tuple[dict[int, str], dict[str, int]]:
    """All splits share the same 77-class Banking77 taxonomy."""
    splits = load_banking77()
    return splits.id_to_label, splits.label_to_id


# ---------- Checkpoint ----------


def _checkpoint_path(split: str, run_id: str) -> Path:
    return _CHECKPOINT_DIR / f"batch_label_{split}_{run_id}.jsonl"


def _write_checkpoint_row(path: Path, idx: int, labeled: LabeledMessage, gold: int) -> None:
    """Append a single labeled row to the JSONL checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"idx": idx, "gold_label_id": gold, **asdict(labeled)}
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _read_checkpoint(path: Path) -> list[Result]:
    """Read all rows from a JSONL checkpoint. Returns [] if file missing."""
    if not path.exists():
        return []
    results: list[Result] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            idx = record.pop("idx")
            gold = record.pop("gold_label_id")
            labeled = LabeledMessage(**record)
            results.append((idx, labeled, gold))
    return results


def _row_cost(labeled: LabeledMessage) -> float:
    return (
        labeled.input_tokens * USD_PER_M_INPUT_TOKENS
        + labeled.output_tokens * USD_PER_M_OUTPUT_TOKENS
    ) / 1_000_000


# ---------- Concurrent labeling with checkpoint ----------


def label_split(
    client: TeacherClient,
    examples: Sequence[tuple[str, int]],
    taxonomy_block: str,
    label_to_id: dict[str, int],
    *,
    checkpoint_path: Path,
    concurrency: int = 5,
    max_cost_usd: float = 5.0,
    skip_indices: set[int] | None = None,
    progress_every: int = 100,
) -> list[Result]:
    """Teacher-label each example concurrently; persist results row-by-row.

    Indices in `skip_indices` are not submitted (used on resume to pick up where
    a prior run left off). Halts cleanly if cumulative cost exceeds `max_cost_usd`.
    """
    skip = skip_indices or set()
    pending = [
        (idx, msg, gold) for idx, (msg, gold) in enumerate(examples) if idx not in skip
    ]
    results: list[Result] = []
    cost_so_far = 0.0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(label_message, client, msg, taxonomy_block, label_to_id): (idx, gold)
            for idx, msg, gold in pending
        }

        for completed_idx, future in enumerate(as_completed(futures)):
            idx, gold = futures[future]
            try:
                labeled = future.result()
            except Exception as e:
                log.error("label_failed", index=idx, error=str(e))
                continue
            results.append((idx, labeled, gold))
            _write_checkpoint_row(checkpoint_path, idx, labeled, gold)
            cost_so_far += _row_cost(labeled)

            done = completed_idx + 1
            if done % progress_every == 0:
                print(f"  labeled {done}/{len(pending)}  cost=${cost_so_far:.4f}")

            if cost_so_far > max_cost_usd:
                log.error(
                    "cost_cap_exceeded",
                    cost_so_far=round(cost_so_far, 4),
                    cap=max_cost_usd,
                )
                executor.shutdown(wait=False, cancel_futures=True)
                break

    return results


# ---------- Output: summary + predictions DF + prompt snapshot ----------


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    xs = sorted(values)
    if not xs:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        k = p * (len(xs) - 1)
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return {"p50": round(pct(0.50), 1), "p95": round(pct(0.95), 1), "p99": round(pct(0.99), 1)}


def compute_summary(
    results: Sequence[Result],
    id_to_label: dict[int, str],
    split: str,
    model_version: str,
    wall_time_s: float,
    run_id: str,
    prompt_version: str,
    prompt_fp: str,
) -> dict[str, Any]:
    """Aggregate per-call telemetry into the summary.json for the run."""
    n = len(results)
    errors = sum(1 for _, lm, _ in results if lm.error is not None)
    input_tokens = sum(lm.input_tokens for _, lm, _ in results)
    output_tokens = sum(lm.output_tokens for _, lm, _ in results)
    cost = sum(_row_cost(lm) for _, lm, _ in results)
    latencies = [lm.latency_ms for _, lm, _ in results]

    class_dist: dict[str, int] = {}
    for _, lm, _ in results:
        if lm.teacher_intent_id == -1:
            continue
        class_dist[lm.teacher_intent_name] = class_dist.get(lm.teacher_intent_name, 0) + 1

    # Sanity-check: against gold, what fraction matches? Not used in training; for ops only.
    matches_gold = sum(1 for _, lm, g in results if lm.teacher_intent_id == g)
    teacher_vs_gold_accuracy = round(matches_gold / n, 4) if n else 0.0

    return {
        "run_id": run_id,
        "split": split,
        "model": model_version,
        "prompt": {"version": prompt_version, "fingerprint": prompt_fp},
        "rows": n,
        "errors": errors,
        "error_rate": round(errors / n, 4) if n else 0.0,
        "teacher_vs_gold_accuracy": teacher_vs_gold_accuracy,
        "tokens": {"input_total": input_tokens, "output_total": output_tokens},
        "cost_usd": round(cost, 4),
        "wall_time_s": round(wall_time_s, 1),
        "latency_ms": _percentiles(latencies),
        "teacher_label_distribution": dict(
            sorted(class_dist.items(), key=lambda kv: -kv[1])
        ),
    }


def build_predictions_dataframe(
    results: Sequence[Result],
    id_to_label: dict[int, str],
    split: str,
    prompt_version: str,
    prompt_fp: str,
) -> pd.DataFrame:
    """One row per labeled example. `split` column is the downstream-filter guardrail."""
    rows = [
        {
            **asdict(lm),
            "gold_label_id": g,
            "gold_label_name": id_to_label[g],
            "correct": lm.teacher_intent_id == g,
            "split": split,
            "prompt_version": prompt_version,
            "prompt_fingerprint": prompt_fp,
        }
        for _, lm, g in results
    ]
    return pd.DataFrame(rows)


def build_prompt_snapshot(valid_intents: list[str], taxonomy_block: str) -> dict[str, Any]:
    """Full record of every input that determined teacher behavior — for reconstruction."""
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


# ---------- GCS upload ----------


def upload_to_gcs(local_path: Path, bucket_name: str, gcs_path: str) -> None:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))


# ---------- CLI ----------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=VALID_SPLITS)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Label only the first N rows (dev/smoke). Default: whole split.",
    )
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help=f"Cost cap. Per-split defaults: {DEFAULT_COST_CAPS}.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Run ID to resume from. Reads its checkpoint and skips already-done rows.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print estimate and exit.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--no-upload", action="store_true", help="Skip GCS upload.")
    parser.add_argument("--output-bucket", default=DEFAULT_OUTPUT_BUCKET)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    split = args.split
    if args.max_cost_usd is None:
        args.max_cost_usd = DEFAULT_COST_CAPS[split]

    print(f"Loading split: {split}")
    df = load_split(split)
    if args.n is not None and args.n < len(df):
        df = df.iloc[: args.n].reset_index(drop=True)
    print(f"  {len(df)} rows")

    id_to_label, label_to_id = load_label_maps()
    taxonomy = format_class_list(id_to_label)
    valid_intents = sorted(id_to_label.values())
    prompt_snapshot = build_prompt_snapshot(valid_intents, taxonomy)
    prompt_fp = prompt_snapshot["prompt_fingerprint"]
    print(f"\nPrompt: version={PROMPT_VERSION}  fingerprint={prompt_fp}")

    avg_in = len(taxonomy) // _CHARS_PER_TOKEN + _AVG_MESSAGE_TOKENS + _PROMPT_OVERHEAD_TOKENS
    avg_out = _AVG_OUTPUT_TOKENS
    est_per_call = (
        avg_in * USD_PER_M_INPUT_TOKENS + avg_out * USD_PER_M_OUTPUT_TOKENS
    ) / 1_000_000
    est_total = est_per_call * len(df)
    print(
        f"\nEstimated cost: ~${est_total:.4f} for {len(df)} rows "
        f"(~{avg_in} in / {avg_out} out per call)"
    )
    print(f"Hard cap: ${args.max_cost_usd:.2f}")

    if args.dry_run:
        print("(dry run; exiting before any API calls)")
        return 0

    if not args.yes:
        confirm = input(f"\nContinue with {len(df)} {split} examples (~${est_total:.4f})? [y/N]: ")
        if confirm.lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    run_id = args.resume or datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    checkpoint_path = _checkpoint_path(split, run_id)
    existing = _read_checkpoint(checkpoint_path)
    skip = {idx for idx, _, _ in existing}
    if skip:
        print(f"\nResuming from checkpoint: {len(skip)} rows already labeled.")
    print(f"Run ID: {run_id}")

    os.environ["MAX_CALLS_PER_RUN"] = str(len(df) + 50)
    TeacherClient.reset_call_count()
    client = from_env()

    examples = [(str(row["text"]), int(row["label"])) for _, row in df.iterrows()]
    remaining = len(df) - len(skip)

    print(f"\nLabeling {remaining} remaining (concurrency={args.concurrency})...")
    start = time.monotonic()
    new_results = label_split(
        client=client,
        examples=examples,
        taxonomy_block=taxonomy,
        label_to_id=label_to_id,
        checkpoint_path=checkpoint_path,
        concurrency=args.concurrency,
        max_cost_usd=args.max_cost_usd,
        skip_indices=skip,
    )
    wall = time.monotonic() - start

    all_results = sorted(existing + new_results, key=lambda r: r[0])
    print(f"\nLabeling complete: {len(all_results)}/{len(df)} successful in {wall:.1f}s")

    summary = compute_summary(
        all_results,
        id_to_label,
        split,
        client._model,
        wall,
        run_id,
        PROMPT_VERSION,
        prompt_fp,
    )
    pred_df = build_predictions_dataframe(
        all_results, id_to_label, split, PROMPT_VERSION, prompt_fp
    )

    local_dir = Path("data") / "teacher_labels" / split / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(local_dir / "labels.parquet", index=False)
    (local_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (local_dir / "prompt_snapshot.json").write_text(json.dumps(prompt_snapshot, indent=2))

    if not args.no_upload:
        print(f"\nUploading to gs://{args.output_bucket}/{args.output_prefix}/{split}/{run_id}/...")
        for filename in ("labels.parquet", "summary.json", "prompt_snapshot.json"):
            gcs_path = f"{args.output_prefix}/{split}/{run_id}/{filename}"
            upload_to_gcs(local_dir / filename, args.output_bucket, gcs_path)
            print(f"  ✓ gs://{args.output_bucket}/{gcs_path}")

    print("\n" + "=" * 60)
    print(f"Split: {split}  Rows: {len(all_results)}/{len(df)}")
    print(f"Cost:        ${summary['cost_usd']:.4f}")
    print(f"Wall time:   {wall:.1f}s")
    print(f"Errors:      {summary['errors']}/{summary['rows']}")
    print(f"Teacher vs gold: {summary['teacher_vs_gold_accuracy']:.1%}")
    p = summary["latency_ms"]
    print(f"Latency:     p50={p['p50']:.0f}ms  p95={p['p95']:.0f}ms  p99={p['p99']:.0f}ms")
    print(f"GCS:         gs://{args.output_bucket}/{args.output_prefix}/{split}/{run_id}/")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
