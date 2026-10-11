#!/bin/bash

WORKSPACE="/petrobr/parceirosbr/home/victor.setti/workspace"
PROJECT_ROOT="$WORKSPACE/Seismic-Transforms"
SCRIPT_PATH="$PROJECT_ROOT/deeprock/deeprock_setti.py"
export SIF="/petrobr/parceirosbr/spfm/singularity/arm64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"


NUM_EXPERIMENTS=3
PREDICTION_HEAD="aspp"
USE_RANDOM="true"
FREEZE="false"
TASKS="salt:tgs"
LR="1e-3"
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

for (( i=0; i<NUM_EXPERIMENTS; i++ )); do
    EXP_NUM=$(printf "%02d" $i)
    EXPERIMENT_NAME="exp_${EXP_NUM}_${BASE_NAME}"

    OUTPUT_DIR="$PROJECT_ROOT/outputs/tgs/$BASE_NAME/$EXPERIMENT_NAME"
    JOBS_OUT_DIR="$OUTPUT_DIR/jobs_out"

    mkdir -p "$JOBS_OUT_DIR"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${EXPERIMENT_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=21
#SBATCH --gpus-per-node=1
#SBATCH --partition=ict-gh200
#SBATCH --account=spfm
#SBATCH --time=03:00:00
#SBATCH --output=${JOBS_OUT_DIR}/%j.out
#SBATCH --error=${JOBS_OUT_DIR}/%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "=== Informações do Job ==="
echo "ID do Job: \$SLURM_JOB_ID"
echo "Nome do Experimento: $EXPERIMENT_NAME"
echo "Seed: $i"
echo "Nós alocados: \$SLURM_JOB_NODELIST"
echo "Data de início: \$(date)"
echo "=========================="

nvidia-smi

singularity exec --nv \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm \
    "$SIF" bash -c "python3 $SCRIPT_PATH $PY_ARGS --seed $i"
EOT

    echo "✅ Submitted: $EXPERIMENT_NAME"
done