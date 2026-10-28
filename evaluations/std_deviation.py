import os
import argparse
import pandas as pd
import numpy as np

base_dir = "/petrobr/parceirosbr/spfm/victor.setti/outputs/deeplab_aspp_pretrain_fullft/"

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--metric", required=True)
args = parser.parse_args()

task = args.task
target_metric = args.metric
dir_path = os.path.join(base_dir, task)

scores = []

# Iterate through all run directories (0, 1, 2, ...)
for run_dir in sorted(os.listdir(dir_path)):
    run_path = os.path.join(dir_path, run_dir)
    
    if not os.path.isdir(run_path):
        continue

    # Point directly to the evaluation metrics file
    eval_metrics_file = os.path.join(run_path, "evaluate", "best_val", "metrics.csv")
    
    if not os.path.exists(eval_metrics_file):
        print(f"Run {run_dir}: Evaluation metrics not found at {eval_metrics_file}")
        continue

    try:
        df = pd.read_csv(eval_metrics_file)
        
        if df.empty:
            print(f"Run {run_dir}: {eval_metrics_file} is empty.")
            continue

        if target_metric not in df.columns:
            print(f"Run {run_dir}: Column '{target_metric}' not found in {eval_metrics_file}.")
            continue

        # Drop NaN values in the target metric column to isolate valid evaluations
        valid_scores = df[target_metric].dropna()
        
        if valid_scores.empty:
            print(f"Run {run_dir}: No valid numeric data in column '{target_metric}'.")
            continue
            
        # Get the value from the last logged line for this metric
        final_score = valid_scores.iloc[-1]
        scores.append(final_score)
        print(f"Run {run_dir} | {target_metric}: {final_score:.4f}")
        
    except Exception as e:
        print(f"Error reading evaluation metrics for Run {run_dir}: {e}")

# Calculate mean and standard deviation across test runs
if scores:
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
    
    print("-" * 40)
    print(f"Target Metric: {target_metric}")
    print(f"Final Test Result (n={len(scores)}): {mean_score:.4f} (+- {std_score:.4f})")
else:
    print("No valid scores found across any evaluation directories.")