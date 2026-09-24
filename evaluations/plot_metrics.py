import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path

def clean_lightning_csv(csv_path):
    """
    PyTorch Lightning's CSVLogger writes train, val, and test metrics on different rows.
    This function groups the data by epoch and averages them into a clean DataFrame.
    """
    df = pd.read_csv(csv_path)
    
    if 'step' in df.columns:
        df = df.drop(columns=['step'])
        
    # Merges alternating train/val rows. Epoch 100 will naturally have NaNs for train/val.
    df_clean = df.groupby('epoch').mean().reset_index()
    return df_clean

def plot_learning_curves(df, plots_dir, exp_name):
    """
    Generates and saves the loss curve and the metrics curves (including final test scores).
    """
    plt.style.use('ggplot')
    
    # Isolate the test data (Epoch 100 in your CSV)
    test_cols = [c for c in df.columns if c.startswith('test_')]
    test_data = df.dropna(subset=test_cols).tail(1) if test_cols else pd.DataFrame()
    
    # Isolate continuous data to prevent Matplotlib from trying to plot NaNs at Epoch 100
    df_train = df.dropna(subset=['train_loss']) if 'train_loss' in df.columns else df
    df_val = df.dropna(subset=['val_loss']) if 'val_loss' in df.columns else df

    # ---------------------------------------------------------
    # 1. Plot Loss Curve (Train vs Val vs Test)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    if 'train_loss' in df.columns and 'val_loss' in df.columns:
        plt.plot(df_train['epoch'], df_train['train_loss'], label='Train Loss', color='tab:blue', linewidth=2)
        plt.plot(df_val['epoch'], df_val['val_loss'], label='Validation Loss', color='tab:red', linewidth=2)
        
        # Plot single Test Loss point if available
        if not test_data.empty and 'test_loss' in test_data.columns:
            test_epoch = test_data['epoch'].values[0]
            test_loss_val = test_data['test_loss'].values[0]
            plt.scatter(test_epoch, test_loss_val, color='black', marker='*', s=220, zorder=5, 
                        edgecolor='white', label=f'Test Loss ({test_loss_val:.4f})')
        
        plt.yscale('log')
        plt.title(f'Loss Curve ({exp_name})', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        loss_plot_path = plots_dir / 'loss_curve.png'
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved Loss plot to: {loss_plot_path}")
    else:
        print("⚠️ Could not find both Train and Val loss columns in the CSV. Skipping loss plot.")
    plt.close()

    # -------------------------------------------------------------
    # 2. Plot Metrics (Validation Curves + Final Test Points)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    metrics_map = {
        'IoU_Standard': ('val_IoU_Standard', 'test_IoU_Standard', 'tab:green'), 
        'TGS_Benchmark': ('val_TGS_Benchmark', 'test_TGS_Benchmark', 'tab:blue'),
        'acc': ('val_acc', 'test_acc', 'tab:orange'), 
        'f1-weighted': ('val_f1-weighted', 'test_f1-weighted', 'tab:purple')
    }
    
    metrics_plotted = False
    test_summary_text = []

    for name, (val_col, test_col, color) in metrics_map.items():
        # 1. Plot continuous Validation line (dropping the NaN at Epoch 100)
        if val_col in df.columns:
            df_metric = df.dropna(subset=[val_col])
            plt.plot(df_metric['epoch'], df_metric[val_col] * 100, label=f'Val {name}', color=color, linewidth=2)
            metrics_plotted = True
        
        # 2. Plot final Test point as a Star (*)
        if not test_data.empty and test_col in test_data.columns:
            test_epoch = test_data['epoch'].values[0]
            test_val = test_data[test_col].values[0] * 100
            plt.scatter(test_epoch, test_val, color=color, marker='*', s=220, edgecolor='black', zorder=5,
                        label=f'Test {name} ({test_val:.1f}%)')
            test_summary_text.append(f"{name}: {test_val:.2f}%")

    if metrics_plotted:
        plt.title(f'Metrics ({exp_name})', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Score (%)', fontsize=14)
        
        # Position the legend outside or adjust loc to prevent overlapping the text box
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True)

        metrics_plot_path = plots_dir / 'val_metrics_curve.png'
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved Metrics plot to: {metrics_plot_path}")
    else:
        print("⚠️ Could not find validation metrics in the CSV. Skipping metrics plot.")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot metrics from PyTorch Lightning CSV logs.")
    parser.add_argument("--exp-name", type=str, required=True, help="Name of the experiment directory")
    parser.add_argument("--group-name", type=str, default=None, help="Name of the parent grouping directory")

    args = parser.parse_args()

    OUT_PROJECT_ROOT = Path("/petrobr/parceirosbr/spfm/victor.setti/outputs/tgs/")
    OUT_PERSONAL_ROOT = Path("/petrobr/parceirosbr/home/victor.setti/workspace/Seismic-Transforms/outputs/tgs/")

    if args.group_name:
        logs_dir = OUT_PROJECT_ROOT / args.group_name / args.exp_name / "logs"
        plots_dir = OUT_PERSONAL_ROOT / args.group_name / args.exp_name / "plots"
    else:
        logs_dir = OUT_PROJECT_ROOT / args.exp_name / "logs"
        plots_dir = OUT_PERSONAL_ROOT / args.exp_name / "plots"

    plots_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(logs_dir.rglob("metrics.csv"))
    
    if not csv_files:
        print(f"❌ Error: Could not find 'metrics.csv' inside {logs_dir}")
        sys.exit(1)
        
    csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"📊 Found metrics file: {csv_path}")
    
    df_clean = clean_lightning_csv(csv_path)
    plot_learning_curves(df_clean, plots_dir, args.exp_name)

if __name__ == "__main__":
    main()