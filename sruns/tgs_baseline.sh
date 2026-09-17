#!/bin/bash

EXPERIMENT_NAME="exp_00_baseline_finetuning"

WORKSPACE="/petrobr/parceirosbr/home/victor.setti/workspace"
PROJECT_ROOT="$WORKSPACE/Seismic-Transforms"

SCRIPT_PATH="$PROJECT_ROOT/scripts/train_tgs.py"
export SIF="/petrobr/parceirosbr/spfm/singularity/arm64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

OUTPUT_DIR="$PROJECT_ROOT/outputs/tgs/$EXPERIMENT_NAME"
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
#SBATCH --time=06:00:00
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
    "$SIF" bash -c "python3 $SCRIPT_PATH --exp-name $EXPERIMENT_NAME"
EOT