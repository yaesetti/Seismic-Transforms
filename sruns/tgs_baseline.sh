#!/bin/bash

NUM_EXPERIMENTS=5
BASE_NAME="baseline_lr1e-3_cosine_wd1e-4_fixed"

WORKSPACE="/petrobr/parceirosbr/home/victor.setti/workspace"
PROJECT_ROOT="$WORKSPACE/Seismic-Transforms"
SCRIPT_PATH="$PROJECT_ROOT/scripts/train_tgs.py"
export SIF="/petrobr/parceirosbr/spfm/singularity/arm64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

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
#SBATCH --time=01:00:00
#SBATCH --output=${JOBS_OUT_DIR}/%j.out
#SBATCH --error=${JOBS_OUT_DIR}/%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "=== Informações do Job ==="
echo "ID do Job: \$SLURM_JOB_ID"
echo "Nós alocados: \$SLURM_JOB_NODELIST"
echo "Flags utilizadas: $FLAGS"
echo "Data de início: \$(date)"
echo "=========================="

nvidia-smi

# Ajustado o terceiro --bind para mapear origem:destino corretamente
singularity exec --nv \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm \
    "$SIF" bash -c "python3 $SCRIPT_PATH --exp-name $EXPERIMENT_NAME --group-name $BASE_NAME"
EOT

    echo "✅ Submitted: $EXPERIMENT_NAME"
done