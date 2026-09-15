import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path

def clean_lightning_csv(csv_path):
    """
    PyTorch Lightning's CSVLogger writes train and val metrics on different rows.
    This function groups the data by epoch and averages the metrics, 
    creating a clean, continuous DataFrame.
    """
    df = pd.read_csv(csv_path)
    
    # Drop the 'step' column as we want to plot by 'epoch'
    if 'step' in df.columns:
        df = df.drop(columns=['step'])
        
    # Group by epoch and mean to merge the train and val rows
    df_clean = df.groupby('epoch').mean().reset_index()
    return df_clean

def plot_learning_curves(df, plots_dir):
    """
    Generates and saves the loss curve and the validation metrics curves.
    """
    # Use a nice style for the plots
    plt.style.use('ggplot')
    
    # ---------------------------------------------------------
    # 1. Plot Loss Curve (Train vs Validation)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Find loss columns (Minerva/Lightning might name them slightly differently depending on your logs)
    train_loss_col = next((col for col in df.columns if 'train' in col.lower() and 'loss' in col.lower()), None)
    val_loss_col = next((col for col in df.columns if 'val' in col.lower() and 'loss' in col.lower()), None)
    
    if train_loss_col and val_loss_col:
        plt.plot(df['epoch'], df[train_loss_col], label='Train Loss', color='tab:blue', linewidth=2)
        plt.plot(df['epoch'], df[val_loss_col], label='Validation Loss', color='tab:red', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        loss_plot_path = plots_dir / 'loss_curve.png'
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved Loss plot to: {loss_plot_path}")
    else:
        print("⚠️ Could not find both Train and Val loss columns in the CSV. Skipping loss plot.")
    plt.close()

    # ---------------------------------------------------------
    # 2. Plot Validation Metrics (IoU, Accuracy, F1)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Define colors for different metrics
    colors = {'val_IoU': 'tab:green', 'val_acc': 'tab:orange', 'val_f1-weighted': 'tab:purple'}
    metrics_plotted = False
    
    for metric, color in colors.items():
        if metric in df.columns:
            # Multiply by 100 to show as percentage
            plt.plot(df['epoch'], df[metric] * 100, label=metric.replace('val_', ''), color=color, linewidth=2)
            metrics_plotted = True

    if metrics_plotted:
        plt.title('Validation Metrics', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Score (%)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        metrics_plot_path = plots_dir / 'val_metrics_curve.png'
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved Metrics plot to: {metrics_plot_path}")
    else:
        print("⚠️ Could not find validation metrics (IoU, acc, f1) in the CSV. Skipping metrics plot.")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot metrics from PyTorch Lightning CSV logs.")
    parser.add_argument("experiment_name", type=str, help="Name of the experiment directory (e.g., exp_01_swav_full_freeze)")
    args = parser.parse_args()

    # Resolve paths based on the workspace structure
    # This assumes plot_metrics.py is inside Seismic-Transforms/scripts/
    project_root = Path(__file__).resolve().parent.parent
    exp_dir = project_root / "outputs" / "tgs_salt" / args.experiment_name
    
    logs_dir = exp_dir / "logs"
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate the metrics.csv file
    # Lightning usually puts it directly in logs_dir or inside a 'version_0' subfolder
    csv_files = list(logs_dir.rglob("metrics.csv"))
    
    if not csv_files:
        print(f"❌ Error: Could not find 'metrics.csv' inside {logs_dir}")
        sys.exit(1)
        
    # If there are multiple (e.g. you resumed training), pick the most recently modified one
    csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"📊 Found metrics file: {csv_path}")
    
    # Process and plot
    df_clean = clean_lightning_csv(csv_path)
    plot_learning_curves(df_clean, plots_dir)

if __name__ == "__main__":
    main()
