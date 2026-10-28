import os
import re
import argparse
import pandas as pd
import numpy as np
from scipy import stats

def extract_scores(base_dir, target_metric):
    """Extracts scores from a given experiment directory, returning a dictionary of {run_number: score}."""
    scores_dict = {}
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return scores_dict

    for run_dir in sorted(os.listdir(base_dir)):
        run_path = os.path.join(base_dir, run_dir)
        
        if not os.path.isdir(run_path):
            continue

        # Point directly to the evaluation metrics file
        metrics_file = os.path.join(run_path, "evaluate", "best_val", "metrics.csv")
        
        if not os.path.exists(metrics_file):
            continue

        # Read metrics and extract the score from the last valid row
        try:
            df = pd.read_csv(metrics_file)
            
            if df.empty or target_metric not in df.columns:
                continue
                
            # Isolate the target column, drop NaNs, and get the final logged value
            val_scores = df[target_metric].dropna()
            
            if not val_scores.empty:
                scores_dict[run_dir] = val_scores.iloc[-1]
                
        except Exception:
            continue
            
    return scores_dict

# Define the paths to your two experiments
base_dir = "/petrobr/parceirosbr/spfm/victor.setti/outputs/deeplab_aspp_pretrain_fullft/"

parser = argparse.ArgumentParser()
parser.add_argument("--task1", required=True)
parser.add_argument("--task2", required=True)
parser.add_argument("--metric", required=True)
args = parser.parse_args()

exp1_dir = os.path.join(base_dir, args.task1)
exp2_dir = os.path.join(base_dir, args.task2)
target_metric = args.metric

# --- Extraction ---
print(f"Extracting scores for {target_metric}...\n")
exp1_scores = extract_scores(exp1_dir, target_metric)
exp2_scores = extract_scores(exp2_dir, target_metric)

# Find common runs (seeds) to ensure a perfectly paired comparison
common_runs = sorted(list(set(exp1_scores.keys()).intersection(set(exp2_scores.keys()))))

if not common_runs:
    print("No common runs found between the two experiments.")
    exit()

# Extract aligned lists for statistical testing
aligned_exp1 = [exp1_scores[r] for r in common_runs]
aligned_exp2 = [exp2_scores[r] for r in common_runs]

# --- Output & Statistics ---
print("Run-by-Run Comparison:")
for r, s1, s2 in zip(common_runs, aligned_exp1, aligned_exp2):
    diff = s2 - s1
    print(f"Run {r}: {args.task1} = {s1:.4f} | {args.task2} = {s2:.4f} | Diff = {diff:+.4f}")

print("\n" + "="*40)
print(f"Total Paired Runs: {len(common_runs)}")

mean1, std1 = np.mean(aligned_exp1), np.std(aligned_exp1, ddof=1)
mean2, std2 = np.mean(aligned_exp2), np.std(aligned_exp2, ddof=1)

print(f"{args.task1}: {mean1:.4f} (+- {std1:.4f})")
print(f"{args.task2}: {mean2:.4f} (+- {std2:.4f})")

# Calculate Paired t-test
if len(common_runs) > 1:
    test_stat, p_value = stats.wilcoxon(aligned_exp1, aligned_exp2)
    print("-" * 40)
    print(f"Wilcoxon p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: STATISTICALLY SIGNIFICANT difference.")
    else:
        print("Result: NO statistically significant difference.")
else:
    print("\nCannot calculate p-value with fewer than 2 paired runs.")