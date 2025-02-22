debug = True
SEED = 42

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, NllbTokenizer, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, DataCollatorForSeq2Seq, BitsAndBytesConfig, HfArgumentParser
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from datasets import Dataset, load_dataset, concatenate_datasets
import evaluate

from math import inf as INF
import numpy as np
import os, sys
import logging
import random
from collections import defaultdict
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
from trl import SFTTrainer, SFTConfig
from argparse import ArgumentParser

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "generating_artificial_dialects/noisers/"))
from main import parse_noise_params, get_noisers, apply_noisers, record_noiser_artifacts, apply_noisers_compose_augment, apply_noisers_compose


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--exp_key", type=str, required=True, help="Experiment identifier")
    parser.add_argument("--hrln", type=str, required=True, help="High-resource language name")
    parser.add_argument("--source_corpus", type=str, default=None, help="HRLN corpus - bitext source side")
    parser.add_argument("--target_corpus", type=str, default=None, help="English corpus - bitext target side")
    parser.add_argument("--hf_dataset_name", type=str, default=None, help="Huggingface dataset name")
    parser.add_argument("--augment_ratio_alpha", type=float, default=1.0, help="Augmentation ratio: 1.0 means all examples are augmented, 0.0 means none are augmented and we are doing fthrln")
    parser.add_argument("--max_lines", type=int, default=1000, help="Number of lines to read from the dataset")
    parser.add_argument("--model_name",type=str, required=True, help="Model to finetune, must be one of 'm2m', 'nllb', 'aya-23-8b'")
    parser.add_argument("--trained_model_path", type=str, default=None, help="Path to trained Aya model")
    parser.add_argument("--lora", action="store_true", help="Whether to add LoRA to the model")
    parser.add_argument("--theta_func_global", type=float, default=0, help="Function word global noise rate")
    parser.add_argument("--theta_content_global", type=float, default=0, help="Content word global noise rate")
    parser.add_argument("--theta_morph_global", type=float, default=0, help="Morphological global noise rate")
    parser.add_argument("--theta_phon", type=float, default=0, help="Phonological noise rate")
    parser.add_argument("--theta_semantic_global", type=float, default=0, help="Semantic noise rate")
    parser.add_argument("--theta_random_char", type=float, default=0, help="Random character noise rate")
    parser.add_argument("--theta_random_word", type=float, default=0, help="Random word noise rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--NUM_ARTLANGS", type=int, default=10, help="Number of artificial languages for augmentation.")
    parser.add_argument("--fixed_set", action="store_true", help="Whether to use a fixed set of artificial languages")
    parser.add_argument("--MODEL_OUTPUT_DIR", type=str, default="checkpoints/", help="Model output directory")
    parser.add_argument("--TRAIN_LOG_DIR", type=str, default="train_logs/", help="Training log directory")

    args = parser.parse_args()
    return args
    

def setup_logger(name, log_dir, level=logging.INFO):
    """To setup as many loggers as you want"""

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"train.log")

    handler = logging.FileHandler(log_file, mode='w')        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def init_noisers_fixed_set(lang, read_file, theta_func_global, theta_content_global, \
    theta_morph_global, theta_phon, \
        theta_semantic_global = 0, theta_random_char = 0, theta_random_word = 0, NUM_ARTLANGS = 10):

    '''
    Initialize noisers for a fixed set of artificial languages. 
    The map of changes is sampled only once for each noiser and then applied to all examples.
    '''
    # We'll have NUM_ARTLANGS noiser classes 
    # Each noiser class will have the same noise parameters but randomly samples
    # a different (fixed) artificial language. 
    # We have a total of NUM_ARTLANGS artificial languages
    noiser_classes = []
    ## The following is for shell. Currently, we are doing -cloud noising.
    # for idx in range(NUM_ARTLANGS):
    #     noise_specs_idx = f"lexical-lang={lang},theta_content_global={theta_content_global},theta_func_global={theta_func_global},text_file=<{read_file}>;morph-lang={lang},theta_morph_global={theta_morph_global},text_file=<{read_file}>;phonological-lang={lang},theta_phon={theta_phon},text_file=<{read_file}>"
    #     all_noise_params_idx = parse_noise_params(noise_specs_idx)
    #     noiser_classes_idx = get_noisers(all_noise_params_idx)
    #     noiser_classes.append(noiser_classes_idx)

    for idx in range(1, NUM_ARTLANGS + 1):
        theta_func_global_idx = theta_func_global * idx / NUM_ARTLANGS
        theta_content_global_idx = theta_content_global * idx / NUM_ARTLANGS
        theta_morph_global_idx = theta_morph_global * idx / NUM_ARTLANGS
        theta_phon_idx = theta_phon * idx / NUM_ARTLANGS
        # theta_semantic_global_idx = theta_semantic_global * idx / NUM_ARTLANGS # GlobalSemanticNoiser not yet implemented
        theta_content_global_idx = theta_content_global * idx / NUM_ARTLANGS
        # theta_random_char_idx = theta_random_char * idx / NUM_ARTLANGS # GlobalRandomCharNoiser not yet implemented
        # theta_random_word_idx = theta_random_word * idx / NUM_ARTLANGS # GlobalRandomWordNoiser not yet implemented
        # noise_specs_idx = f"lexical_aug-lang={lang},theta_content_global={theta_content_global},theta_func_global={theta_func_global},text_file=<{read_file}>;morph_aug-lang={lang},theta_morph_global={theta_morph_global},text_file=<{read_file}>;phonological_aug-lang={lang},theta_phon={theta_phon},text_file=<{read_file}>;random_char_aug-lang={lang},theta_random_char={theta_random_char};random_word_aug-lang={lang},theta_random_word={theta_random_word},text_file=<{read_file}>"
        noise_specs_idx = f"lexical-lang={lang},theta_content_global={theta_content_global_idx},\
theta_func_global={theta_func_global_idx},text_file=<{read_file}>;morph-lang={lang},\
theta_morph_global={theta_morph_global_idx},text_file=<{read_file}>;\
phonological-lang={lang},theta_phon={theta_phon_idx},text_file=<{read_file}>"
        all_noise_params_idx = parse_noise_params(noise_specs_idx)
        noiser_classes_idx = get_noisers(all_noise_params_idx)
        noiser_classes.append(noiser_classes_idx)


    return noiser_classes



def init_noisers(lang, read_file, theta_func_global, theta_content_global, \
    theta_morph_global, theta_phon, \
        theta_semantic_global = 0, theta_random_char = 0, theta_random_word = 0, NUM_ARTLANGS = 10):

    '''
    This is for the (default) case where we sample a new artificial language for each example.
    '''

    # We'll have NUM_ARTLANGS noiser classes uniformly distributed in the hypersphere
    # Every time we apply a noiser class, it results in a new artificial language.
    noiser_classes = []
    for idx in range(1, NUM_ARTLANGS + 1):
        theta_func_global_idx = theta_func_global * idx / NUM_ARTLANGS
        theta_content_global_idx = theta_content_global * idx / NUM_ARTLANGS
        theta_morph_global_idx = theta_morph_global * idx / NUM_ARTLANGS
        theta_phon_idx = theta_phon * idx / NUM_ARTLANGS
        theta_semantic_global_idx = theta_semantic_global * idx / NUM_ARTLANGS
        theta_content_global_idx = theta_content_global * idx / NUM_ARTLANGS
        theta_random_char_idx = theta_random_char * idx / NUM_ARTLANGS
        theta_random_word_idx = theta_random_word * idx / NUM_ARTLANGS

        noise_specs_idx = f"lexical_aug-lang={lang},theta_content_global={theta_content_global_idx},\
theta_func_global={theta_func_global_idx},text_file=<{read_file}>;morph_aug-lang={lang},\
theta_morph_global={theta_morph_global_idx},text_file=<{read_file}>;\
phonological_aug-lang={lang},theta_phon={theta_phon_idx},text_file=<{read_file}>;\
semantic_aug-lang={lang},theta_semantic_global={theta_semantic_global_idx},text_file=<{read_file}>;\
random_char_aug-lang={lang},theta_random_char={theta_random_char_idx};\
random_word_aug-lang={lang},theta_random_word={theta_random_word_idx},text_file=<{read_file}>"
        all_noise_params_idx = parse_noise_params(noise_specs_idx)
        noiser_classes_idx = get_noisers(all_noise_params_idx)
        noiser_classes.append(noiser_classes_idx)

    return noiser_classes


def map_augment_artifical_langs(examples):
    global noiser_classes, NUM_ARTLANGS, augment_ratio_alpha, fixed_set
    aug_source = []
    aug_target = []
    for source, target in zip(examples["source"], examples["target"]):
        if random.random() < augment_ratio_alpha:
            # Sample a noiser class
            noiser_class_idx = random.randint(0, NUM_ARTLANGS - 1)
            noiser_class = noiser_classes[noiser_class_idx]
            compose_func = apply_noisers_compose if fixed_set else apply_noisers_compose_augment
            noised_source = compose_func(source, noiser_class, verbose=False)
            aug_source.append(noised_source)
        else:
            aug_source.append(source)
        aug_target.append(target)

    return {"source": aug_source, "target": aug_target}


def map_add_prompt(examples):
    # prompt = "Translate into English: "
    return {"source": [f"Translate into English: \n{ex}" for ex in examples["source"]], "target": examples["target"]}

def map_formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['source'])):
        text = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{example['source'][i]}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{example['target'][i]}<|END_OF_TURN_TOKEN|>"
        output_texts.append(text)
    return output_texts

    
def map_tokenize(examples, max_length = 512):
        # HF doesn't support separate source and target tokenizers, 
        # so we assume it's the same tokenizer for now.
        if "text" in examples:
            inputs = [ex for ex in examples["text"]]
            model_inputs = tokenizer(
                inputs, max_length=max_length, truncation=True, padding="max_length"
            )
            return model_inputs
        else:
            inputs = [ex for ex in examples["source"]]
            targets = [ex for ex in examples["target"]]
            model_inputs = tokenizer(
                inputs, text_target=targets, max_length=max_length, truncation=True, padding="max_length"
            )
        return model_inputs


def prepare_and_tokenize(dataset, add_prompt = False):

    logger.info("DATA AUGMENTATION")
    dataset = dataset.map(map_augment_artifical_langs, batched = True)

    if add_prompt:
        dataset = dataset.map(map_add_prompt, batched = True, remove_columns=["source", "target"])

    logger.info(f"AUGMENTED EXAMPLES: {len(dataset)}")
    if debug:
        logger.info("Examples:")
        logger.info(dataset[:10])
    
    if not add_prompt:
        logger.info("STARTING TOKENIZING")
        # default batch size is 1000
        
        dataset = dataset.map(map_tokenize, batched = True, \
                    remove_columns=["source", "target"] if not add_prompt else ["text"])
        
        logger.info("DONE TOKENIZING!")

    if debug:
        logger.info("Examples:")
        logger.info(dataset[0])
    
    dataset = dataset.with_format("torch")
    return dataset

def get_dataset_from_hf(dataset_name, hrln, max_lines, tokenizer, max_length = 512, add_prompt = False):
    # Stream it so that we don't load the entire dataset into memory
    if dataset_name == "allenai/nllb":
        iso3_to_nllbcode = {
            "hin": "hin_Deva",
            "tur": "tur_Latn",
            "arb": "arb_Arab",
            "ind": "ind_Latn",
            "ita": "ita_Latn",
            "hat": "hat_Latn"
            }
        src_lang = iso3_to_nllbcode[hrln]
        tgt_lang = "eng_Latn"
        l1, l2 = sorted([src_lang, tgt_lang])
        datadir_name = f"{l1}-{l2}"

        dataset = load_dataset(dataset_name, datadir_name, split="train", streaming=True)
        dataset = Dataset.from_dict({"source":[elem["translation"][src_lang] for elem in dataset], "target":[elem["translation"][tgt_lang] for elem in dataset]})

    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)

    dataset = dataset.select(range(min(max_lines * 10, len(dataset)))) # just to make sure we select randomly from a big enough subset
    dataset = dataset.shuffle(seed=SEED)
    dataset = dataset.select(range(min(max_lines, len(dataset))))
    logger.info(f"LOADED DATASET SIZE: {len(dataset)}")
    logger.info("Examples:")
    logger.info(dataset[0])

    dataset = prepare_and_tokenize(dataset, add_prompt = add_prompt)
    return dataset

def get_dataset_from_files(SOURCE_FILES, TARGET_FILES, max_lines, tokenizer, max_length = 512, add_prompt = False):
    
    logger.info("Loading Datasets...")
    dataset = load_dataset("text", data_files={"source": SOURCE_FILES, \
        "target": TARGET_FILES})
    
    # Create a new dataset that has source and target as columns
    dataset = Dataset.from_dict({"source": dataset["source"]["text"], "target": dataset["target"]["text"]})
    
    dataset = dataset.shuffle(seed=SEED)
    dataset = dataset.select(range(min(max_lines, len(dataset))))
    logger.info(f"LOADED DATASET SIZE: {len(dataset)}")
    logger.info("Examples:")
    logger.info(dataset[0])

    dataset = prepare_and_tokenize(dataset, add_prompt = add_prompt)

    return dataset

def get_dataset_from_multiple_files(SOURCE_FILES, TARGET_FILES, max_lines, tokenizer, max_length = 512, add_prompt = False):
    
    logger.info("Loading Datasets...")
    max_lines_per_file = max_lines // len(SOURCE_FILES)
    datasets = []
    for source_file, target_file in zip(SOURCE_FILES, TARGET_FILES):
        print(f"Loading {source_file} and {target_file}")
        dataset = load_dataset("text", data_files={"source": source_file, \
            "target": target_file})
        dataset = Dataset.from_dict({"source": dataset["source"]["text"], "target": dataset["target"]["text"]})
        dataset = dataset.shuffle(seed=SEED)
        dataset = dataset.select(range(min(max_lines_per_file, len(dataset))))
        print(f"LOADED DATASET SIZE: {len(dataset)}")
        datasets.append(dataset)
    
    # Concatenate all datasets  
    dataset = concatenate_datasets(datasets)

    # Format of dataset: {"source": List[str], "target": List[str]}
    logger.info(f"LOADED DATASET SIZE: {len(dataset)}")
    logger.info("Examples:")
    logger.info(dataset[0])

    dataset = prepare_and_tokenize(dataset, add_prompt = add_prompt)

    return dataset



def get_splits(dataset):
    '''Get train, validation, and test splits'''
    # Split into train, validation, and test - 95%-4%-1% split (since we don't actually want to test on the test set)
    train_size = int(0.95 * len(dataset))
    val_size = int(0.04 * len(dataset))
    test_size = int(0.01 * len(dataset))
    train_devtest = dataset.train_test_split(test_size=val_size)
    dev_test = train_devtest["test"].train_test_split(test_size=test_size)
    train_dataset = train_devtest["train"]
    val_dataset = dev_test["train"]
    test_dataset = dev_test["test"]
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def get_val_sets(hrln, add_prompt = False):
    '''
    Construct validation test suite
    '''
    hrln2crls = {
        "hin": ["hin_Deva", "bho_Deva", "mag_Deva"],
        "tur": ["tur_Latn", "tuk_Latn", "crh_Latn"],
        "ita": ["ita_Latn", "vec_Latn", "scn_Latn"],
        "ind": ["ind_Latn", "plt_Latn", "pag_Latn"],
        "arb": ["arb_Arab", "acm_Arab", "acq_Arab"],
        "hat": ["gcf", "hat", "lou"]
        }

    langs = hrln2crls[hrln]
    val_dataset = {}
    for lang in langs:
        if "hat" in hrln:
            source_file = f"/home/nrobin38/kreyol-mt-naacl24/OfficialTestSetsFromRaj/data_from_raj/local_all_public/{lang}--eng/test.cleaned.{lang}"
            target_file = f"/home/nrobin38/kreyol-mt-naacl24/OfficialTestSetsFromRaj/data_from_raj/local_all_public/{lang}--eng/test.cleaned.eng"
        else:
            source_file = f"/export/b08/nbafna1/data/flores200_dataset/dev/{lang}.dev"
            target_file = "/export/b08/nbafna1/data/flores200_dataset/dev/eng_Latn.dev"
        with open(source_file, "r") as f:
            source_sentences = f.readlines()
        with open(target_file, "r") as f:
            target_sentences = f.readlines()
        source_sentences = source_sentences[:100]
        target_sentences = target_sentences[:100]
        dataset = Dataset.from_dict({"source": source_sentences, "target": target_sentences})
        if add_prompt:
            dataset = dataset.map(map_add_prompt, batched = True)
            # dataset = dataset.map(map_formatting_prompts_func, batched = True)
        
        if not add_prompt:
            dataset = dataset.map(map_tokenize, batched = True, \
                    remove_columns=["source", "target"] if not add_prompt else ["text"])
        
        dataset = dataset.with_format("torch")
        val_dataset[lang] = dataset
    return val_dataset

def compute_metrics(pred):
    '''Compute BLEU score'''
    global tokenizer, bleu

    # Get predictions
    predictions = pred.predictions
    labels = pred.label_ids
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU
    if sum([len(pred.split()) for pred in predictions]) == 0:
        return {'bleu': 0.0, \
                'brevity_penalty': 0, \
                'length_ratio': 0, 'translation_length': 0, 'reference_length': 0}
    

    bleu_metric = bleu.compute(predictions = predictions, references = labels)

    # Remove precisions
    bleu_metric = {k: v for k, v in bleu_metric.items() if k != "precisions"}

    return bleu_metric

def main():
    global tokenizer, bleu, noiser_classes, NUM_ARTLANGS, logger, augment_ratio_alpha, fixed_set
    
    args = parse_args()
    exp_key = args.exp_key
    hrln = args.hrln
    source_corpus = args.source_corpus
    target_corpus = args.target_corpus
    hf_dataset_name = args.hf_dataset_name
    NUM_ARTLANGS = args.NUM_ARTLANGS
    fixed_set = args.fixed_set
    augment_ratio_alpha = args.augment_ratio_alpha
    max_lines = args.max_lines
    theta_func_global = args.theta_func_global
    theta_content_global = args.theta_content_global
    theta_morph_global = args.theta_morph_global
    theta_phon = args.theta_phon
    theta_semantic_global = args.theta_semantic_global
    theta_random_char = args.theta_random_char
    theta_random_word = args.theta_random_word
    model_name = args.model_name
    trained_model_path = args.trained_model_path
    lora = args.lora
    epochs = args.epochs
    batch_size = args.batch_size
    MODEL_OUTPUT_DIR = args.MODEL_OUTPUT_DIR
    TRAIN_LOG_DIR = args.TRAIN_LOG_DIR
        
    OUTPUT_DIR = f"{MODEL_OUTPUT_DIR}/{hrln}/{exp_key}"
    LOG_DIR = f"{TRAIN_LOG_DIR}/{hrln}/{exp_key}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Set up logger
    logger = setup_logger(f'finetune_{model_name}_alaug', f"/export/b08/nbafna1/projects/dialectical-robustness-mt/train_logs/{hrln}/{exp_key}")

    if model_name == "nllb":
        iso3_to_nllbcode = {
            "hin": "hin_Deva",
            "tur": "tur_Latn",
            "arb": "arb_Arab",
            "rus": "rus_Cyrl",
            "ind": "ind_Latn",
            "ita": "ita_Latn",
            "hat": "hat_Latn"
        }
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=iso3_to_nllbcode[hrln], tgt_lang="eng_Latn")
        # Add LoRA to the model
        print(f"LoRA: {lora}")
        if lora:
            logger.info("Adding LoRA to the model...")
            print("Adding LoRA to the model...")
            lora_config = LoraConfig(
                target_modules="all-linear",      
                r=8,                          
                lora_alpha=1,                
                lora_dropout=0.1              
            )
            model = get_peft_model(model, lora_config)
    
    elif model_name == "m2m" or model_name == "m2mL":
        iso3_to_m2mcode = {
            "hin": "hi",
            "tur": "tr",
            "arb": "ar",
            "rus": "ru",
            "ind": "id",
            "ita": "it",
            "hat": "ht",
        }
        model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
        tokenizer.src_lang = iso3_to_m2mcode[hrln]
        tokenizer.tgt_lang = "en"

        # Add LoRA to the model
        print(f"LoRA: {lora}")
        if lora:
            logger.info("Adding LoRA to the model...")
            print("Adding LoRA to the model...")
            lora_config = LoraConfig(
                target_modules="all-linear",      
                r=8,                          
                lora_alpha=1,                
                lora_dropout=0.1              
            )
            model = get_peft_model(model, lora_config)
        
        if trained_model_path:
            raise NotImplementedError("Continued finetuning of trained model not implemented for m2m model")

    elif model_name == "aya-23-8b":
        QUANTIZE_4BIT = True
        USE_GRAD_CHECKPOINTING = True
        TRAIN_MAX_SEQ_LENGTH = 512
        USE_FLASH_ATTENTION = True
        GRAD_ACC_STEPS = 2
        MODEL_NAME = "CohereForAI/aya-23-8b"
        quantization_config = None
        if QUANTIZE_4BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        attn_implementation = None
        if USE_FLASH_ATTENTION:
            attn_implementation="flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        if trained_model_path:
            logger.info(f"Loading trained model from {trained_model_path}")
            print(f"Loading trained model from {trained_model_path}")
            model.load_adapter(trained_model_path)

    # DA data
    SOURCE_FILES = source_corpus.split(",")
    TARGET_FILES = target_corpus.split(",")
    print(f"Source files: {SOURCE_FILES}")
    print(f"Target files: {TARGET_FILES}")

    # Put the read_file to the last file in the list
    noiser_init_func = init_noisers_fixed_set if fixed_set else init_noisers
    noiser_classes = noiser_init_func(hrln, SOURCE_FILES[-1], \
        theta_func_global = theta_func_global, theta_content_global = theta_content_global, \
        theta_morph_global = theta_morph_global, theta_phon = theta_phon, theta_semantic_global = theta_semantic_global, \
        theta_random_char = theta_random_char, theta_random_word = theta_random_word, NUM_ARTLANGS = NUM_ARTLANGS)

    add_prompt = True if model_name == "aya-23-8b" else False # This adds a prompt to the input text, which is necessary for the aya-23-8b model

    if hf_dataset_name:
        dataset = get_dataset_from_hf(hf_dataset_name, hrln, max_lines = max_lines, tokenizer = tokenizer, add_prompt = add_prompt)
    else:
        dataset = get_dataset_from_multiple_files(SOURCE_FILES, TARGET_FILES, max_lines = max_lines, tokenizer = tokenizer, add_prompt = add_prompt)
        
    train_dataset, val_dataset, test_dataset = get_splits(dataset)

    
    val_dataset_all = {}
    val_dataset_all["artificial_langs"] = val_dataset
    # Val data on real CRLs from FloRes200 
    ## If you want to add this, fill in the source and target filepaths in the get_val_sets function
    ## and uncomment the following line
    # val_dataset_all.update(get_val_sets(hrln, add_prompt = add_prompt))
    

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
    # Metric
    bleu = evaluate.load("bleu")

    # Initialize Seq2SeqTrainer
    logger.info("Initializing trainer...")
    
    resume_from_checkpoint = False
    only_eval = False

    training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    resume_from_checkpoint=resume_from_checkpoint,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True if model_name == "aya-23-8b" else False,
    optim="paged_adamw_32bit" if model == "aya-23-8b" else "adamw_torch",
    overwrite_output_dir=False,
    num_train_epochs=epochs,
    learning_rate=1e-5,
    fp16=False,
    bf16=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.05,
    group_by_length=True,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    predict_with_generate=True,
    generation_max_length=40, # defaults to model config max_length
    report_to="tensorboard",
    evaluation_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    eval_steps=1500, 
    logging_steps=50,
    save_steps=3000, 
    load_best_model_at_end=False,
    save_total_limit=3
    )

    if model_name in {"m2m", "m2mL", "nllb"}:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset_all,
            tokenizer=tokenizer,
            data_collator= data_collator,
            compute_metrics=compute_metrics,
            # callbacks=[CustomTensorboardCallback],
        )   
    elif model_name in {"aya-23-8b"}:

        peft_config = LoraConfig(
            lora_alpha=32,
            r=32,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        tokenizer.padding_side = "right" # This is necessary for the aya-23-8b model
        # This is the training arguments for the SFTTrainer, doesn't have generation_max_length, predict_with_generate, etc.
        training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        resume_from_checkpoint=resume_from_checkpoint,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True if model_name == "aya-23-8b" else False,
        optim="paged_adamw_32bit" if model == "aya-23-8b" else "adamw_torch",
        overwrite_output_dir=False,
        num_train_epochs=epochs,
        learning_rate=1e-5,
        fp16=False,
        bf16=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.05,
        group_by_length=True,
        weight_decay=0.01,
        logging_dir=LOG_DIR,
        report_to="tensorboard",
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=1500, 
        logging_steps=500,
        save_steps=3000, 
        load_best_model_at_end=False,
        save_total_limit=3
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset_all,
            peft_config=peft_config,
            max_seq_length=TRAIN_MAX_SEQ_LENGTH,
            tokenizer=tokenizer,
            args=training_args,
            formatting_func=map_formatting_prompts_func,
        )

    if not only_eval:
        # Check if CUDA is available
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        logger.info("STARTING TRAINING")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        logger.info("SAVING MODEL")
        trainer.model.save_pretrained(OUTPUT_DIR)

    # Get performance and labels on test set
    if test_dataset:

        logger.info("STARTING EVALUATION")
        test_results = trainer.predict(test_dataset)
        test_metrics = test_results.metrics
        predictions = test_results.predictions
        labels = test_results.label_ids 
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode into text
        inputs = tokenizer.batch_decode(test_dataset["input_ids"], skip_special_tokens=True)
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Log examples
        logger.info("logger examples...")
        for i in range(len(predictions[:10])):
            logger.info("Example {}: ".format(i))
            logger.info("Input: {}".format(inputs[i]))
            logger.info("Prediction: {}".format(predictions[i]))
            logger.info("Label: {}".format(labels[i]))
        # Log metrics
        logger.info("logger metrics...")
        logger.info("Test metrics: {}".format(test_metrics))


        logger.info("DONE EVALUATION")



if __name__ == "__main__":

    main()