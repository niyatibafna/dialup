#!/bin/bash

#SBATCH --job-name=eval_cloud    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-node=1                # Total number of gpus
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --exclude=c18,c11
#SBATCH --mem=40G                # Total memory allocated
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-54             # job array index
#SBATCH --output=slurm_logs_eval/aya_ftalaug_eval_cloud_%a.out   # output file name
#SBATCH --error=slurm_logs_eval/aya_ftalaug_eval_cloud_%a.out    # error file name


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

declare -a hrl_lrl_pairs
# Add pairs for each hrl and its lrls
hrl_lrl_pairs+=("hin hne_Deva" "hin bho_Deva" "hin mag_Deva" "hin mai_Deva" "hin hin_Deva") # 0-4
hrl_lrl_pairs+=("tur tur_Latn" "tur uzn_Latn" "tur tuk_Latn" "tur azj_Latn" "tur crh_Latn") # 5-9
hrl_lrl_pairs+=("ita spa_Latn" "ita fra_Latn" "ita por_Latn" "ita ita_Latn" "ita ron_Latn" "ita glg_Latn" "ita cat_Latn" "ita oci_Latn" "ita ast_Latn" "ita lmo_Latn" "ita vec_Latn" "ita scn_Latn" "ita srd_Latn" "ita fur_Latn" "ita lij_Latn") # 10-24
hrl_lrl_pairs+=("ind ind_Latn" "ind jav_Latn" "ind sun_Latn" "ind smo_Latn" "ind mri_Latn" "ind ceb_Latn" "ind zsm_Latn" "ind tgl_Latn" "ind ilo_Latn" "ind fij_Latn" "ind plt_Latn" "ind pag_Latn") # 25-36
# hrl_lrl_pairs+=("arb bag" "arb cai" "arb dam" "arb doh" "arb fes" "arb jer" "arb kha" \
# "arb msa" "arb riy" "arb san" "arb tri" "arb tun") # 37-48 # This is for evaluating on the MADAR dataset
hrl_lrl_pairs+=("arb arb_Arab" "arb acm_Arab" "arb acq_Arab" "arb aeb_Arab" "arb ajp_Arab" "arb apc_Arab" "arb ars_Arab" "arb ary_Arab" "arb arz_Arab") # 37-45
hrl_lrl_pairs+=("hat hat" "hat gcf" "hat mart1259" "hat acf" "hat gcr" "hat lou" "hat mfe" "hat rcf" "hat crs") # 46-54

hrl_lrl_pair=${hrl_lrl_pairs[$SLURM_ARRAY_TASK_ID]}
hrln=$(echo $hrl_lrl_pair | cut -d ' ' -f 1)
crl=$(echo $hrl_lrl_pair | cut -d ' ' -f 2)

model_name="aya-23-8b"

theta_content_global=0.001
theta_morph_global=0.3
theta_phon=0.07
theta_func_global=0.8

max_lines=100000

MODEL_OUTPUT_DIR="/export/b08/nbafna1/projects/dialectical-robustness-mt/checkpoints"
TRAIN_LOG_DIR="/export/b08/nbafna1/projects/dialectical-robustness-mt/train_logs"

exp_name="ftalaug-cloud"
exp_key="${model_name}_${exp_name}_${hrln}_tf-${theta_func_global}_tc-${theta_content_global}_tm-${theta_morph_global}_tp-${theta_phon}_max_lines-${max_lines}"
echo "Experiment key: $exp_key"

model_path="$MODEL_OUTPUT_DIR/$hrln/$exp_key/"

flores_dir="/export/b08/nbafna1/data/flores200_dataset/"
kreyolmt_dir="/home/nrobin38/kreyol-mt-naacl24/OfficialTestSetsFromRaj/data_from_raj/local_all_public/"

mt_outputs_dir="/export/b08/nbafna1/projects/dialectical-robustness-mt/outputs/mt_outputs_fullset/"
results_dir="/export/b08/nbafna1/projects/dialectical-robustness-mt/results_fullset/"

mkdir -p $mt_outputs_dir
mkdir -p $results_dir


python evaluate_mtd.py \
--exp_key $exp_key \
--hrln $hrln \
--crl $crl \
--model_name $model_name \
--model_path $model_path \
--flores_dir $flores_dir \
--kreyolmt_dir $kreyolmt_dir \
--lora \
--batch_size 4 \
--mt_outputs_dir $mt_outputs_dir \
--results_dir $results_dir 