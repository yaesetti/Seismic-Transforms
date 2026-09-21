#!/bin/bash

EXPERIMENT_NAME="exp_00_baseline_lr1e-4_no-scheduler_BS256"

WORKSPACE="/petrobr/parceirosbr/home/victor.setti/workspace"
PROJECT_ROOT="$WORKSPACE/Seismic-Transforms"

SCRIPT_PATH="$PROJECT_ROOT/evaluations/plot_metrics.py"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

OUTPUT_DIR="$PROJECT_ROOT/outputs/tgs/$EXPERIMENT_NAME"
PLOTS_DIR="$OUTPUT_DIR/plots"

mkdir -p "$PLOTS_DIR"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=plot_${EXPERIMENT_NAME}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=00:03:00
#SBATCH --output=${PLOTS_DIR}/plot_%j.out
#SBATCH --error=${PLOTS_DIR}/plot_%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Allocated nodes: \$SLURM_JOB_NODELIST"
nvidia-smi

singularity exec --nv \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm \
    "$SIF" bash -c "python3 $SCRIPT_PATH $EXPERIMENT_NAME"
EOT