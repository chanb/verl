#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=07:00:00
#SBATCH --mem=500GB
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:l40s:4
#SBATCH --array=1-1
#SBATCH --output=/home/chanb/scratch/logs/hint_without_realizability/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
module load apptainer/1.4.5

apptainer run --nv -C -W $SLURM_TMPDIR -B ~/research/hint_without_realizability:/workspace -B ~/scratch/datasets:/datasets -B ~/scratch/verl_checkpoints:/verl_checkpoints ~/research/hint_without_realizability/verl.sif bash /workspace/verl/cc_scripts/run_ppo.sh
