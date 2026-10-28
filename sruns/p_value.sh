#!/bin/bash

WORKSPACE="/petrobr/parceirosbr/home/victor.setti/workspace"
PROJECT_ROOT="$WORKSPACE/Seismic-Transforms"

# Point to your new aggregation script
SCRIPT_PATH="$PROJECT_ROOT/evaluations/p_value.py"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

# Create a central folder for the SLURM logs
BASE_NAME="deeplab_aspp_pretrain_fullft"

TASK1="salt/tgs"
TASK2="salt/tgs+crop"

METRIC="epoch_test_tgs-iou"

P_VALUE_DIR="$PROJECT_ROOT/outputs/$BASE_NAME/$TASK2/p_value"
mkdir -p "$P_VALUE_DIR"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=p_value${BASE_NAME}_${TASK1}_${TASK2}
#SBATCH --nodes=1
#SBATCH --partition=cpu_amd
#SBATCH --account=spfm
#SBATCH --time=00:05:00
#SBATCH --output=${P_VALUE_DIR}/jobs_out/%j.out
#SBATCH --error=${P_VALUE_DIR}/jobs_out/%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Allocated nodes: \$SLURM_JOB_NODELIST"

singularity exec \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm \
    "$SIF" bash -c "python3 $SCRIPT_PATH --task1 $TASK1 --task2 $TASK2 --metric $METRIC"
EOT

echo "✅ Submitted p_value for $BASE_NAME for $TASK1 and $TASK2"