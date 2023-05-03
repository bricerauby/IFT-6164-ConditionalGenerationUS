#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --account=def-jeproa
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --job-name=train_genUlm
#SBATCH --output=output_dir/GanUlm/%j-%x.out

cd ~/IFT-6164-ConditionalGenerationUS

module load python/3.9
module load cuda
module load httpproxy
# module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirementsCCStyleGan.txt

cp -rv ~/scratch/data/data_CGenULM/patchesIQ_small_shuffled $SLURM_TMPDIR/
cp -rv ~/scratch/data/data_CGenULM/cGenUlmSimu $SLURM_TMPDIR/
cp -v ~/scratch/GanCGenPatches.h5 $SLURM_TMPDIR/
cp -v ~/scratch/data/data_CGenULM/BaselineCGenPatches.h5 $SLURM_TMPDIR/
python trainStyleGanUlmConditionned.py
