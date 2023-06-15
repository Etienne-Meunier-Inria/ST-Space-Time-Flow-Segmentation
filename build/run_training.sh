#!/bin/bash
#OAR -l {gpu_model NOT IN ('GeForce RTX 2080 Ti', 'Tesla K80', 'Tesla P100')}/gpu_device=1,walltime=500:00:00
#OAR -n Run-GitHub-SpaceTime
#OAR -O /net/serpico-fs2/emeunier/Logs/ssgrp_job.%jobid%.output
#OAR -E /net/serpico-fs2/emeunier/Logs/ssgrp_job.%jobid%.output
set -xv

. /etc/profile.d/modules.sh

module load spack/singularity

cd /temp_dd/igrida-fs1/emeunier/code/Space-Time-Flow-Segmentation


singularity exec -B /net:/net --nv $Dataria/Singularity/Temporal.sif \
            python3 model_train.py --path_save_model model230323\
                                   --binary_method fair\
                                   --base_dir /net/serpico-fs2/emeunier \
                                   --data_file DataSplit/DAVIS_D16Split\
                                   --gpus 1
