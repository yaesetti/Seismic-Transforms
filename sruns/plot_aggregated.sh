#!/bin/bash

# You only need to define the group name now
GROUP_NAME="baseline_lr1e-3_cosine_wd1e-4_fixed"

WORKSPACE="/petrobr/parceirosbr/home/victor.setti/workspace"
PROJECT_ROOT="$WORKSPACE/Seismic-Transforms"

# Point to your new aggregation script
SCRIPT_PATH="$PROJECT_ROOT/evaluations/plot_aggregated_metrics.py"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

# Create a central folder for the SLURM logs
PLOTS_DIR="$PROJECT_ROOT/outputs/tgs/$GROUP_NAME/aggregate_plots"
mkdir -p "$PLOTS_DIR"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=agg_plot_${GROUP_NAME}
#SBATCH --nodes=1
#SBATCH --partition=cpu_amd
#SBATCH --account=spfm
#SBATCH --time=00:05:00
#SBATCH --output=${PLOTS_DIR}/agg_plot_%j.out
#SBATCH --error=${PLOTS_DIR}/agg_plot_%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Allocated nodes: \$SLURM_JOB_NODELIST"

singularity exec \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm \
    "$SIF" bash -c "python3 $SCRIPT_PATH --group-name $GROUP_NAME"
EOT

echo "✅ Submitted aggregated plot job for group: $GROUP_NAME"