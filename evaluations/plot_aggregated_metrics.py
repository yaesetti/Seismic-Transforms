import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path

def clean_lightning_csv(csv_path):
    """
    Groups the data by epoch and averages alternating rows into a clean DataFrame.
    """
    df = pd.read_csv(csv_path)
    if 'step' in df.columns:
        df = df.drop(columns=['step'])
    return df.groupby('epoch').mean().reset_index()

def plot_aggregated_curves(combined_df, test_df, plots_dir, group_name):
    """
    Plots the mean curve with a shaded standard deviation region for validation metrics,
    and includes test scores with standard deviation error bars.
    """
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    # Calculate mean and std grouped by epoch
    grouped = combined_df.groupby('epoch').agg(['mean', 'std'])
    
    metrics = {
        'val_TGS_Benchmark': ('TGS Benchmark', 'tab:blue'),
        'val_IoU_Standard': ('Standard IoU', 'tab:green')
    }
    
    for col, (label, color) in metrics.items():
        if col in grouped.columns.levels[0]:
            epochs = grouped.index
            mean = grouped[col]['mean'] * 100
            std = grouped[col]['std'] * 100
            
            # Plot validation mean curve
            plt.plot(epochs, mean, label=f'Val {label} (Mean)', color=color, linewidth=2)
            # Plot shaded standard deviation region
            plt.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
            
    # Handle test metrics (plot as points with error bars)
    if not test_df.empty:
        for val_col, (label, color) in metrics.items():
            test_col = val_col.replace('val_', 'test_')
            if test_col in test_df.columns:
                test_mean = test_df[test_col].mean() * 100
                test_std = test_df[test_col].std() * 100
                test_epoch = test_df['epoch'].max()
                
                plt.errorbar(
                    test_epoch, test_mean, yerr=test_std, 
                    fmt='*', color=color, ecolor='black', capsize=5, 
                    markersize=15, markeredgecolor='black', zorder=5,
                    label=f'Test {label} ({test_mean:.1f} ± {test_std:.1f}%)'
                )

    plt.title(f'Aggregated Validation & Test Metrics\n({group_name})', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Score (%)', fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True)
    
    plot_path = plots_dir / 'aggregated_metrics_std.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Aggregated plot to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated metrics with standard deviation across a group.")
    parser.add_argument("--group-name", type=str, required=True, help="Name of the parent grouping directory")

    args = parser.parse_args()

    OUT_PROJECT_ROOT = Path("/petrobr/parceirosbr/spfm/victor.setti/outputs/tgs/")
    OUT_PERSONAL_ROOT = Path("/petrobr/parceirosbr/home/victor.setti/workspace/Seismic-Transforms/outputs/tgs/")

    # Search for all CSVs inside the parent group folder
    group_logs_dir = OUT_PROJECT_ROOT / args.group_name
    plots_dir = OUT_PERSONAL_ROOT / args.group_name / "aggregate_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(group_logs_dir.rglob("metrics.csv"))
    
    if not csv_files:
        print(f"❌ Error: No 'metrics.csv' files found inside {group_logs_dir}")
        sys.exit(1)
        
    print(f"📊 Found {len(csv_files)} experiment(s) in group '{args.group_name}'. Aggregating...")
    
    all_dfs = []
    test_dfs = []
    
    for csv_path in csv_files:
        df = clean_lightning_csv(csv_path)
        
        # Separate test data (usually Epoch 100 with valid test_ cols)
        test_cols = [c for c in df.columns if c.startswith('test_')]
        if test_cols:
            test_data = df.dropna(subset=test_cols).tail(1)
            test_dfs.append(test_data)
            
        # Keep only validation epochs for continuous curves
        val_data = df.dropna(subset=['val_IoU_Standard', 'val_TGS_Benchmark'], how='all')
        all_dfs.append(val_data)
        
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    
    plot_aggregated_curves(combined_df, combined_test_df, plots_dir, args.group_name)

if __name__ == "__main__":
    main()