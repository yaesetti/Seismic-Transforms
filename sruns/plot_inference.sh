#!/bin/bash

# Change this to whichever sample index you want to look at
SAMPLE_INDEX=5

GROUP_NAME="baseline_lr1e-3_cosine_wd1e-4_fixed"
EXPERIMENT_NAME="exp_09_baseline_lr1e-3_cosine_wd1e-4_fixed"

WORKSPACE="/petrobr/parceirosbr/home/victor.setti/workspace"
PROJECT_ROOT="$WORKSPACE/Seismic-Transforms"
SCRIPT_PATH="$PROJECT_ROOT/evaluations/plot_inference.py"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

PLOTS_DIR="$PROJECT_ROOT/outputs/tgs/$GROUP_NAME/$EXPERIMENT_NAME/plots"
mkdir -p "$PLOTS_DIR"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=infer_${SAMPLE_INDEX}
#SBATCH --nodes=1
#SBATCH --partition=cpu_amd
#SBATCH --account=spfm
#SBATCH --time=00:05:00
#SBATCH --output=${PLOTS_DIR}/infer_%j.out
#SBATCH --error=${PLOTS_DIR}/infer_%j.err

cd "\$SLURM_SUBMIT_DIR"

singularity exec \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm \
    "$SIF" bash -c "python3 $SCRIPT_PATH --exp-name $EXPERIMENT_NAME --group-name $GROUP_NAME --sample-idx $SAMPLE_INDEX"
EOT

echo "✅ Submitted Inference plot for sample $SAMPLE_INDEX"