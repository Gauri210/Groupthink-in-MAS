"""
Merge all results/raw/run_*.jsonl files and compute the headline metrics.

Usage (from src/):
    python analyze_results.py
Outputs go to results/processed/: combined_results.csv, summary_by_condition.csv,
summary_by_category.csv, per_question_comparison.csv
"""
import glob
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "raw")
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "processed")
CONDITIONS = ["solo_baseline", "group_deliberation", "group_with_contrarian"]

# 1. Load every run file (one JSON record per line)
records = []
for path in sorted(glob.glob(os.path.join(RAW_DIR, "run_*.jsonl"))):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # blank or corrupted line
            rec["source_file"] = os.path.basename(path)
            records.append(rec)
df = pd.DataFrame(records)

# 2. Drop failed runs, then keep the latest file's copy of each (question, condition)
if "error" in df:
    df = df[df["error"].isna()]
df = (df.sort_values("source_file")
        .drop_duplicates(["question_id", "condition"], keep="last")
        .sort_values(["question_id", "condition"]))

# 3. Add helper columns and drop the bulky transcript
df["no_consensus"] = df["final_answer"].isna()
df["num_rounds"] = df["history"].apply(lambda h: len({x["round"] for x in h}))
df = df.drop(columns=["history", "error"], errors="ignore")
df[["is_correct", "flipped", "rescued"]] = df[["is_correct", "flipped", "rescued"]].astype(bool)

print(f"{len(df)} rows after de-duplication (expected 150)")

# 4. Summaries: the mean of a True/False column is its rate
metrics = ["is_correct", "flipped", "rescued", "no_consensus"]
by_condition = df.groupby("condition")[metrics].mean().reindex(CONDITIONS)
by_condition.insert(0, "n", df.groupby("condition").size())
by_category = df.pivot_table(index="category", columns="condition",
                             values="is_correct", aggfunc="mean")[CONDITIONS]

# 5. Per-question view: did each condition get it right?
per_q = df.pivot(index="question_id", columns="condition", values="is_correct")[CONDITIONS]
per_q.insert(0, "category", df.drop_duplicates("question_id").set_index("question_id")["category"])
solo, group, contrarian = (per_q[c] for c in CONDITIONS)
cross = {
    "solo right, group wrong": int((solo & ~group).sum()),
    "group right, solo wrong": int((~solo & group).sum()),
    "contrarian beat group": int((contrarian & ~group).sum()),
    "contrarian lost to group": int((~contrarian & group).sum()),
}

# 6. Save and print
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(os.path.join(OUT_DIR, "combined_results.csv"), index=False)
by_condition.to_csv(os.path.join(OUT_DIR, "summary_by_condition.csv"))
by_category.to_csv(os.path.join(OUT_DIR, "summary_by_category.csv"))
per_q.to_csv(os.path.join(OUT_DIR, "per_question_comparison.csv"))

pd.options.display.float_format = "{:.1%}".format
print("\nBy condition:\n", by_condition)
print("\nAccuracy by category:\n", by_category)
print("\nCross-condition:", cross)