#!/bin/bash
#
#SBATCH --job-name=LLM_Attention_Job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=youremail@address
#SBATCH --output=./attention_event/scripts/slurm_out/slurm%j.out
#SBATCH --gres=gpu:2

# Activate the virtual environment
source ./attention_event/llmenv/bin/activate

# Set the PyTorch CUDA memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# module load python/3.11 skipped since the virtual env has python already
module load cuda/11.1.74

set -x
text_tested=$1        # first input parameter, eg: SecretLifeWalterMitty.txt
check_punct=$2     # second input parameter, eg: 0 or 1 for whether attention to punctuation

python ./attention_event/scripts/attention_event_seg.py "$text_tested" "$check_punct"

exit # tell python to exit
