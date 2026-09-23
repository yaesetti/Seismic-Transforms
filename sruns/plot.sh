#!/bin/bash

NUM_EXPERIMENTS=5
BASE_NAME="baseline_lr1e-3_cosine_wd1e-4_fixed"

WORKSPACE="/petrobr/parceirosbr/home/victor.setti/workspace"
PROJECT_ROOT="$WORKSPACE/Seismic-Transforms"
SCRIPT_PATH="$PROJECT_ROOT/evaluations/plot_metrics.py"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

for (( i=0; i<NUM_EXPERIMENTS; i++)); do

    EXP_NUM=$(printf "%02d" $i)
    EXPERIMENT_NAME="exp_${EXP_NUM}_${BASE_NAME}"

    OUTPUT_DIR="$PROJECT_ROOT/outputs/tgs/$BASE_NAME/$EXPERIMENT_NAME"
    PLOTS_DIR="$OUTPUT_DIR/plots"

    mkdir -p "$PLOTS_DIR"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=plot_${EXPERIMENT_NAME}
#SBATCH --nodes=1
#SBATCH --partition=cpu_amd
#SBATCH --account=spfm
#SBATCH --time=00:05:00
#SBATCH --output=${PLOTS_DIR}/plot_%j.out
#SBATCH --error=${PLOTS_DIR}/plot_%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Allocated nodes: \$SLURM_JOB_NODELIST"

singularity exec \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm \
    "$SIF" bash -c "python3 $SCRIPT_PATH --exp-name $EXPERIMENT_NAME --group-name $BASE_NAME"
EOT

    echo "✅ Submitted plot: $EXPERIMENT_NAME"
done