"""Error analysis for a validate_on_golden run.

Loads predictions.parquet from a run directory and surfaces:
  - top confusion pairs (gold → teacher) by frequency
  - worst-performing gold classes
  - 3 example messages per top confusion pair (so you can eyeball whether the
    teacher's answer is defensible or genuinely wrong)

Use this BEFORE iterating on the prompt — many "errors" on Banking77 are
ambiguous cases where the teacher's answer is at least as defensible as the
gold label. Knowing that ratio reframes what "raise accuracy" actually means.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_RUNS_ROOT = Path("data") / "golden_validation"


def _latest_run(root: Path) -> Path:
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    if not runs:
        raise SystemExit(f"No runs found under {root}")
    return runs[-1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a specific run directory (defaults to most recent)",
    )
    parser.add_argument("--top-pairs", type=int, default=20)
    parser.add_argument("--examples-per-pair", type=int, default=3)
    parser.add_argument("--worst-classes", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = args.run_dir or _latest_run(DEFAULT_RUNS_ROOT)

    df = pd.read_parquet(run_dir / "predictions.parquet")
    print(f"Run: {run_dir.name}")
    if "prompt_version" in df.columns:
        print(f"Prompt: v{df['prompt_version'].iloc[0]}  fp={df['prompt_fingerprint'].iloc[0]}")
    print(f"Total: {len(df)}  correct: {df['correct'].sum()}  acc: {df['correct'].mean():.1%}")

    errors = df[~df["correct"]].copy()
    parse_errors = (errors["teacher_intent_id"] == -1).sum()
    miss_rate = len(errors) / len(df)
    print(f"Misses: {len(errors)} ({miss_rate:.1%})  of which parse-errors: {parse_errors}")

    real_errors = errors[errors["teacher_intent_id"] != -1]

    print(f"\n{'=' * 70}")
    print(f"Top {args.top_pairs} confusion pairs (gold -> teacher)")
    print("=" * 70)
    pairs = Counter(
        zip(real_errors["gold_label_name"], real_errors["teacher_intent_name"], strict=True)
    )
    for (gold, pred), count in pairs.most_common(args.top_pairs):
        print(f"\n[{count:2d}x] {gold}  ->  {pred}")
        examples = real_errors[
            (real_errors["gold_label_name"] == gold) & (real_errors["teacher_intent_name"] == pred)
        ].head(args.examples_per_pair)
        for _, row in examples.iterrows():
            print(f"     · {row['text']}")

    print(f"\n{'=' * 70}")
    print(f"Worst {args.worst_classes} gold classes by accuracy")
    print("=" * 70)
    by_class = df.groupby("gold_label_name").agg(n=("correct", "size"), correct=("correct", "sum"))
    by_class["acc"] = by_class["correct"] / by_class["n"]
    worst = by_class.sort_values(["acc", "n"], ascending=[True, False]).head(args.worst_classes)
    for cls, row in worst.iterrows():
        print(f"  {row['acc']:>5.0%}  ({int(row['correct'])}/{int(row['n'])})  {cls}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
