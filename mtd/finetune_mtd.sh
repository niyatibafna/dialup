#!/bin/bash

#SBATCH --job-name=aya_ftalaug_cloud    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-node=1                # Total number of gpus
#SBATCH --partition=gpu-a100          # Name of the partition
#SBATCH --account=a100acct
#SBATCH --mem=40G                # Total memory allocated
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-5              # job array index
#SBATCH --output=slurm_logs/aya_ftalaug_cloud_%a.out   # output file name
#SBATCH --error=slurm_logs/aya_ftalaug_cloud_%a.out    # error file name

echo "### Running $SLURM_JOB_NAME ###"

cd ${SLURM_SUBMIT_DIR}

echo "HOSTNAME: $(hostname)"
echo
echo CUDA in ENV:
env | grep CUDA
echo

nvidia-smi

module purge
module load conda
conda --version
# module load cuda/10.2
module load cuda/12.1
nvcc --version

# Set your conda environment
source /home/$USER/.bashrc
conda info --envs

which python
. "/home/nbafna1/miniconda3/etc/profile.d/conda.sh" && conda deactivate && conda activate sandbox
which python

set -x
cd "/export/b08/nbafna1/projects/dialectical-robustness-mt/"

hrlns=("hin" "tur" "ind" "arb" "ita" "hat")
hrln=${hrlns[$SLURM_ARRAY_TASK_ID]}

model_name="aya-23-8b"

theta_content_globals=(0.001 0.001 0.001 0.001 0.001 0.001)
theta_morph_globals=(0.3 0.3 0.3 0.3 0.3 0.3)
theta_phons=(0.07 0.07 0.07 0.07 0.07 0.07)
theta_func_globals=(0.8 0.8 0.8 0.8 0.8 0.8)
theta_content_global=${theta_content_globals[$SLURM_ARRAY_TASK_ID]}
theta_morph_global=${theta_morph_globals[$SLURM_ARRAY_TASK_ID]}
theta_phon=${theta_phons[$SLURM_ARRAY_TASK_ID]}
theta_func_global=${theta_func_globals[$SLURM_ARRAY_TASK_ID]}

max_lines=100000
source_corpora=("/export/b08/nbafna1/data/wikimatrix/en-hi/WikiMatrix.en-hi.hi" "/export/b08/nbafna1/data/wikimatrix/en-tr/WikiMatrix.en-tr.tr" "/export/b08/nbafna1/data/wikimatrix/en-id/WikiMatrix.en-id.id" "/export/b08/nbafna1/data/wikimatrix/ar-en/WikiMatrix.ar-en.ar" "/export/b08/nbafna1/data/wikimatrix/en-it/WikiMatrix.en-it.it" "/home/nrobin38/kreyol-mt-naacl24/OfficialTrainSetsFromRaj/hat--eng/train.cleaned.truncated100k.hat")
target_copora=("/export/b08/nbafna1/data/wikimatrix/en-hi/WikiMatrix.en-hi.en" "/export/b08/nbafna1/data/wikimatrix/en-tr/WikiMatrix.en-tr.en" "/export/b08/nbafna1/data/wikimatrix/en-id/WikiMatrix.en-id.en" "/export/b08/nbafna1/data/wikimatrix/ar-en/WikiMatrix.ar-en.en" "/export/b08/nbafna1/data/wikimatrix/en-it/WikiMatrix.en-it.en" "/home/nrobin38/kreyol-mt-naacl24/OfficialTrainSetsFromRaj/hat--eng/train.cleaned.truncated100k.eng")
source_corpus=${source_corpora[$SLURM_ARRAY_TASK_ID]}
target_corpus=${target_copora[$SLURM_ARRAY_TASK_ID]}

MODEL_OUTPUT_DIR="/export/b08/nbafna1/projects/dialectical-robustness-mt/checkpoints"
TRAIN_LOG_DIR="/export/b08/nbafna1/projects/dialectical-robustness-mt/train_logs"

exp_name="ftalaug-cloud"
exp_key="${model_name}_${exp_name}_${hrln}_tf-${theta_func_global}_tc-${theta_content_global}_tm-${theta_morph_global}_tp-${theta_phon}_max_lines-${max_lines}"
echo "Experiment key: $exp_key"

python finetune_mtd.py \
--exp_key $exp_key \
--hrln $hrln \
--source_corpus $source_corpus \
--target_corpus $target_corpus \
--max_lines $max_lines \
--theta_func_global $theta_func_global \
--theta_content_global $theta_content_global \
--theta_morph_global $theta_morph_global \
--theta_phon $theta_phon \
--model_name $model_name \
--lora \
--epochs 1 \
--batch_size 8 \
--MODEL_OUTPUT_DIR $MODEL_OUTPUT_DIR \
--TRAIN_LOG_DIR $TRAIN_LOG_DIR
