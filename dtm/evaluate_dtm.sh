#!/usr/bin/env bash

#SBATCH --job-name=eval_dtm_approach        # create a short name for your job
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=2                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-node=1                   # Total number of gpus
#SBATCH --partition=gpu                     # Name of the partition
#SBATCH --gres=gpu:1
#SBATCH --mem=20G                           # Total memory allocated
#SBATCH --time=02:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --array=0-43                        # job array index
#SBATCH --output=eval_%a.out                # output file name
#SBATCH --error=eval_%a.out                 # error file name

echo "### Running $SLURM_JOB_NAME ###"

cd ${SLURM_SUBMIT_DIR}

echo "HOSTNAME: $(hostname)"

module purge
module load miniforge

conda activate noisers

cd /scratch/ec5ug/noisers/dialup/dtm/

declare -a hrl_lrl_pairs
# Add pairs for each hrl and its lrls
hrl_lrl_pairs+=("hin hne_Deva" "hin bho_Deva" "hin mag_Deva" "hin mai_Deva") # 0-3
hrl_lrl_pairs+=("tur uzn_Latn" "tur tuk_Latn" "tur azj_Latn" "tur crh_Latn") # 4-7
hrl_lrl_pairs+=("ita spa_Latn" "ita fra_Latn" "ita por_Latn" "ita ron_Latn" "ita glg_Latn" "ita cat_Latn" "ita oci_Latn" "ita ast_Latn" "ita lmo_Latn" "ita vec_Latn" "ita scn_Latn" "ita srd_Latn" "ita fur_Latn" "ita lij_Latn") # 8-21
hrl_lrl_pairs+=("ind jav_Latn" "ind sun_Latn" "ind smo_Latn" "ind mri_Latn" "ind ceb_Latn" "ind zsm_Latn" "ind tgl_Latn" "ind ilo_Latn" "ind fij_Latn" "ind plt_Latn" "ind pag_Latn") # 22-32
# hrl_lrl_pairs+=("arb bag" "arb cai" "arb dam" "arb fes" "arb jer" "arb kha" "arb riy" "arb san" "arb tun") # 33-41
hrl_lrl_pairs+=("arb acm_Arab" "arb acq_Arab" "arb aeb_Arab" "arb ajp_Arab" "arb apc_Arab" "arb ars_Arab" "arb ary_Arab" "arb arz_Arab") # 33-40
hrl_lrl_pairs+=("hat acf" "hat crs" "hat mfe") # 41-43

hrl_lrl_pair=${hrl_lrl_pairs[$SLURM_ARRAY_TASK_ID]}
hrln=$(echo $hrl_lrl_pair | cut -d ' ' -f 1)
crl=$(echo $hrl_lrl_pair | cut -d ' ' -f 2)

echo "$hrln"

bilingual_lexicon_path="/sfs/weka/scratch/ec5ug/noisers/dialup/dtm/lexicons"

flores_dir="/sfs/weka/scratch/ec5ug/noisers/flores_devtest/"
kreyolmt_dir="/sfs/weka/scratch/ec5ug/noisers/OfficialTestSetsFromRaj/data_from_raj/local_all_public"

mt_outputs_dir="/scratch/ec5ug/noisers/dialup/dtm/test_outputs"
results_dir="/scratch/ec5ug/noisers/dialup/dtm/test_results"

models=("aya-23-8b" "m2mL")
denoise_funcs=("all" "content" "functional") 

for model_name in "${models[@]}"; do
    for denoise_func in "${denoise_funcs[@]}"; do
        exp_key="${model_name}_${denoise_func}"
        echo "Experiment key: $exp_key"
        
        python evaluate_dtm.py \
        --exp_key $exp_key \
        --hrln $hrln \
        --crl $crl \
        --model_name $model_name \
        --bilingual_lexicon_path $bilingual_lexicon_path \
        --flores_dir $flores_dir \
        --kreyolmt_dir $kreyolmt_dir \
        --denoise_func $denoise_func \
        --batch_size 4 \
        --mt_outputs_dir $mt_outputs_dir \
        --results_dir $results_dir
    done
done

conda deactivate