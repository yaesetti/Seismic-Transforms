#!/bin/bash

apptainer exec  --nv -B /home/shared/deeprock_datasets/ --env PYTHONPATH=$(pwd) /home/shared/singularity/amd64/deeprock/slim/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif python -u test_deeprock.py