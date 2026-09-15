#!/bin/bash

# ==============================================================================
# 1. EXPERIMENT SETUP (Change this to the experiment you want to plot)
# ==============================================================================
EXPERIMENT_NAME="exp_00_baseline"

# ==============================================================================
# 2. PATHS & DIRECTORIES
# ==============================================================================
WORKSPACE="/petrobr/parceirosbr/home/victor.setti/workspace"
PROJECT_ROOT="$WORKSPACE/Seismic-Transforms"

# Pointing to the new plotting script
SCRIPT_PATH="$PROJECT_ROOT/scripts/plot_metrics.py"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

# Set up the output directory for this job's logs
OUTPUT_DIR="$PROJECT_ROOT/outputs/tgs_salt/$EXPERIMENT_NAME"
JOBS_OUT_DIR="$OUTPUT_DIR/jobs_out"

mkdir -p "$JOBS_OUT_DIR"

# ==============================================================================
# 3. SLURM SUBMISSION
# ==============================================================================
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=plot_${EXPERIMENT_NAME}
#SBATCH --nodes=1
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=00:10:00
#SBATCH --output=${JOBS_OUT_DIR}/plot_%j.out
#SBATCH --error=${JOBS_OUT_DIR}/plot_%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Running plot_metrics.py for experiment: $EXPERIMENT_NAME"

# Singularity execution (No --nv needed since we don't need GPUs just to plot)
singularity exec \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm \
    "$SIF" bash -c "python3 $SCRIPT_PATH $EXPERIMENT_NAME"

echo "Plotting finished!"
EOT