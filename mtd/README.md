## Introduction

M-->D is a technique to induce dialectal robustness in an MT model.
Given some HRL that the MT model supports, we perform data augmentation simulating dialectal variation, and finetune the model with it.

## Supported languages

Our noisers support Hindi (`hi`), Arabic (`ar`), Indonesian (`id`), Turkish (`tr`), Italian (`it`), and Haitian (`ht`), German (`de`), English (`en`), Russian (`ru`), Spanish (`es`), French (`fr`).
See our paper for results on how well this techniques works for the first six languages.


## Finetuning

In our finetuning script (`finetune_mtd.py`), we perform data augmentation and LoRA finetune `Aya-23-8B` and `M2M100` (1.2B). Note that we only finetune on augmented data (not the original data).

Here are the options for running this script.

- **Experiment, data**
    - `exp_key`: Something to identify your experiment.
    - `hrln`: HRL code. Should be supported by our noisers (and by the MT model). Currently, our M2M code assumes the target language is English, for the purpose of setting the tokenizer target token.
    - `source_corpus`: Line-separated bitext, source side. This is also used by the noisers as raw data in the source for training chargram model.
    - `target_corpus`: Line-separated bitext, target side.
    - `hf_dataset_name`: If you want to use a HuggingFace dataset instead, set this with the dataset name.
    - `max_lines`: Number of (randomly selected) lines to use.
    - `TRAIN_LOG_DIR`: Training logs.
- **Noising params**: See `generating_artificial_dialects/` for more details on what these noisers are doing, good defaults, and tuning. 
    - `augment_ratio_alpha`: The fraction of the data to augment. 
    - `theta_func_global`: Lexical funtion word noiser parameter.
    - `theta_content_global`: Lexical content word noiser parameter.
    - `theta_morph_global`: Morphological noiser parameter.
    - `theta_phon`: Phonological noiser parameter.
    - `theta_semantic_global`: Semantic noiser. This is supported by a subset of languages.
    - `theta_random_char`: Random character perturber.
    - `theta_random_word`: Random lexical word perturber.
    - `fixed_set`: Whether to do fixed set noising. See our notes for an explanation of the difference.
    - `NUM_ARTLANGS`: Number of radii to sample from, or in the case of fixed_set noising, the number of artificial dialects to synthesize. Beyond our default (10), we found increasing this doesn't really matter.
- **Model and training params**
    - `model_name`: Must be one of `m2m`, `nllb`, `aya-23-8b`.
    - `lora`: Whether to LoRA finetune.
    - `epochs`
    - `batch_size`
    - `MODEL_OUTPUT_DIR`: Where to save checkpoints.

## Evaluation
In our experiments, we evaluate on 
- KreyòlMT (for Haitian Creole-related languages): This is from [Kreyòl-MT: Building MT for Latin American, Caribbean and Colonial African Creole Languages](https://aclanthology.org/2024.naacl-long.170/)
- FloRes, for everything else.

Evaluate with `evaluate_mtd.py`, with the following options:
- `exp_key`: Experiment key
- `hrln`: HRL language code. This is only needed to set the tokenizer source token in case the model does not support the source LRL.
- `crl`: Code of the source dialect/closely-related language we're translating from. In our experiments, the target language was always English.
- `model_name`: Same as above.
- `model_path`: Path to finetuned model checkpoint.
- `lora`: Whether the model was finetuned with LoRA.
- `flores_dir`, `kreyolmt`: Paths to evaluation dataset directories.
- `mt_outputs_dir`: Store the translations.
- `batch_size`
- `prompting_strategy`: We have a few options for prompting Aya. The default (`nolangname`) is `Translate into English: `


