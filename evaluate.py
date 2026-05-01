"""
Grading script — evaluates submission.csv against ground truth.

Usage:
    python evaluate.py --submission_file submission.csv --test_dir <path_to_test_dir>
"""

import argparse
import pandas as pd
from pathlib import Path


def compute_metrics(df: pd.DataFrame, label: str = "") -> dict:
    total   = len(df)
    correct = df["correct"].sum()
    errors  = (df["predicted"].astype(str) == "ERROR").sum()

    print(f"\n{'='*52}")
    if label:
        print(f"  {label}")
    print(f"  Total samples  : {total}")
    print(f"  Correct        : {correct}")
    print(f"  Errors         : {errors}")
    print(f"  Accuracy       : {correct/total:.2%}")

    if "true" in df.columns:
        per_option = (
            df[df["correct"]].groupby("true").size()
            / df.groupby("true").size()
        ).rename("accuracy_per_option")
        print(f"\n  Per-option accuracy:")
        print(per_option.to_string())
    print(f"{'='*52}")

    return {
        "total":    total,
        "correct":  int(correct),
        "errors":   int(errors),
        "accuracy": round(correct / total, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="GNR Project Grading Script")
    parser.add_argument("--submission_file", type=str, required=True,
                        help="Path to submission.csv")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Path to test dir containing test.csv with ground truth")
    args = parser.parse_args()

    # Load submission
    if not Path(args.submission_file).exists():
        print(f"[ERROR] Submission file not found: {args.submission_file}")
        return

    submission = pd.read_csv(args.submission_file)
    print(f"Loaded submission: {len(submission)} rows")
    print(f"Columns: {submission.columns.tolist()}")
    print(submission.head())

    # Load ground truth if available
    gt_path = None
    if args.test_dir:
        gt_path = Path(args.test_dir) / "test.csv"

    if gt_path and gt_path.exists():
        gt = pd.read_csv(gt_path)
        print(f"\nLoaded ground truth: {len(gt)} rows")

        # Merge on image_name
        merged = submission.merge(gt, on="image_name", suffixes=("_pred", "_gt"))

        # Normalize: ensure both are strings for comparison
        merged["predicted"] = merged["predicted"].astype(str).str.strip()
        merged["true"]      = merged["option"].astype(str).str.strip()
        merged["correct"]   = merged["predicted"] == merged["true"]

        compute_metrics(merged, label="Final Evaluation")

        # Save detailed results
        merged[["id", "image_name", "predicted", "true", "correct"]].to_csv(
            "grading_results.csv", index=False
        )
        print("Detailed results saved -> grading_results.csv")
    else:
        # No ground truth — just show distribution
        print("\nNo ground truth found — showing prediction distribution:")
        print(submission["predicted"].value_counts().sort_index())
        print(f"\nTotal predictions: {len(submission)}")
        print(f"Errors: {(submission['predicted'].astype(str) == 'ERROR').sum()}")


if __name__ == "__main__":
    main()
