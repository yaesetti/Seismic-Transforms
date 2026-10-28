#!/bin/bash

WORKSPACE="/petrobr/parceirosbr/home/victor.setti/workspace"
PROJECT_ROOT="$WORKSPACE/Seismic-Transforms"
SCRIPT_PATH="$PROJECT_ROOT/deeprock/deeprock_setti.py"
export SIF="/petrobr/parceirosbr/spfm/singularity/arm64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"


NUM_EXPERIMENTS=10
PREDICTION_HEAD="aspp"
USE_RANDOM="false"
FREEZE="false"
TASKS="facies:aicrowd_seamai+vflip"
LR="1e-4"
STAGES="train+evaluate"

PY_ARGS="--prediction-head $PREDICTION_HEAD --tasks $TASKS --lr $LR --stages $STAGES"

if [ "$USE_RANDOM" = "true" ]; then
    RAND_STR="random"
    PY_ARGS="$PY_ARGS --random"
else
    RAND_STR="pretrain"
fi

if [ "$FREEZE" = "true" ]; then
    FREEZE_STR="freeze"
    PY_ARGS="$PY_ARGS --freeze"
else
    FREEZE_STR="fullft"
fi

if [ "$LR" = "1e-4" ]; then
    LR_STR=""
else
    LR_STR="_lr=${LR}"
fi

BASE_NAME="deeplab_${PREDICTION_HEAD}_${RAND_STR}_${FREEZE_STR}${LR_STR}"

SAFE_TASK_NAME="${TASKS//:/\/}"

for (( i=0; i<NUM_EXPERIMENTS; i++ )); do
    EXPERIMENT_NUMBER="$i"

    OUTPUT_DIR="$PROJECT_ROOT/outputs/$BASE_NAME/$SAFE_TASK_NAME/$EXPERIMENT_NUMBER"
    JOBS_OUT_DIR="$OUTPUT_DIR/jobs_out"

    mkdir -p "$JOBS_OUT_DIR"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${TASKS}_${EXPERIMENT_NUMBER}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=21
#SBATCH --gpus-per-node=1
#SBATCH --partition=ict-gh200
#SBATCH --account=spfm
#SBATCH --time=12:00:00
#SBATCH --output=${JOBS_OUT_DIR}/%j.out
#SBATCH --error=${JOBS_OUT_DIR}/%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "==== Job Informations ===="
echo "Job ID: \$SLURM_JOB_ID"
echo "Run Number / Seed: $EXPERIMENT_NUMBER"
echo "Task: $TASKS"
echo "Allocated Nodes: \$SLURM_JOB_NODELIST"
echo "Start Date: \$(date)"
echo "=========================="

nvidia-smi

singularity exec --nv \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm \
    "$SIF" bash -c "python3 $SCRIPT_PATH $PY_ARGS --seed $i"
EOT

    echo "✅ Submitted: ${SAFE_TASK_NAME}_${EXPERIMENT_NUMBER}"
done