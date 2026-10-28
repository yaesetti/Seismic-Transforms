#!/bin/bash

WORKSPACE="/petrobr/parceirosbr/home/victor.setti/workspace"
PROJECT_ROOT="$WORKSPACE/Seismic-Transforms"

# Point to your new aggregation script
SCRIPT_PATH="$PROJECT_ROOT/evaluations/std_deviation.py"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

# Create a central folder for the SLURM logs
BASE_NAME="deeplab_aspp_pretrain_fullft"
TASKS="binary_channel_identification/cigchannel"

METRIC="epoch_test_iou_jaccard"

STD_DIR="$PROJECT_ROOT/outputs/$BASE_NAME/$TASKS/std_deviation"
mkdir -p "$STD_DIR"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=std_deviation${BASE_NAME/TASKS}
#SBATCH --nodes=1
#SBATCH --partition=cpu_amd
#SBATCH --account=spfm
#SBATCH --time=00:05:00
#SBATCH --output=${STD_DIR}/jobs_out/%j.out
#SBATCH --error=${STD_DIR}/jobs_out/%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Allocated nodes: \$SLURM_JOB_NODELIST"

singularity exec \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm \
    "$SIF" bash -c "python3 $SCRIPT_PATH --task $TASKS --metric $METRIC"
EOT

echo "✅ Submitted std deviation for $BASE_NAME/$TASKS"