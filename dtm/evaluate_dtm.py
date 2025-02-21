import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, NllbTokenizer, \
    BitsAndBytesConfig
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import evaluate
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os, sys
import json
from argparse import ArgumentParser
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from collections import defaultdict

from denoisers import Denoiser
sys.path.append(os.path.join(os.path.dirname(__file__), "utils/"))
from flores_code_to_langname import flores_code_to_langname, get_crls, flores_code_to_hrln

def parse_args():r
    parser = ArgumentParser()
    parser.add_argument("--exp_key", type=str, required=True, help="Experiment key")
    parser.add_argument("--hrln", type=str, required=True, help="High-resource language name (e.g. ita)")
    parser.add_argument("--crl", type=str, required=True, help="CRL language name (e.g. scn_Latn)")
    parser.add_argument("--model_name",type=str, required=True, help="Model to finetune, must be one of 'm2m', 'nllb', 'aya'")
    parser.add_argument("--denoise_func", type=str, default=None, help="Type of denoising function to use, must be one of 'functional', 'content', 'all'")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--lora", action="store_true", help="Use LoRA model")
    parser.add_argument("--bilingual_lexicon_path", type=str, default="../lexicons", help="Path to lexicons")
    parser.add_argument("--flores_dir", type=str, default=None, help="Path to FloRes data")
    parser.add_argument("--kreyolmt_dir", type=str, default=None, help="Path to KreyolMT data, for evaluating on Haitian")
    parser.add_argument("--madar_dir", type=str, default=None, help="Path to MADAR data, for evaluating on Arabic dialects")
    parser.add_argument("--mt_outputs_dir", type=str, required=True, help="Path to save MT outputs")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--prompting_strategy", type=str, default="nolangname", help="Prompting strategy for Aya-23-8b")

    return parser.parse_args()

args = parse_args()
exp_key = args.exp_key
hrln = args.hrln
crl = args.crl
model_name = args.model_name
denoise_func = args.denoise_func
model_path = args.model_path
lora = args.lora
bilingual_lexicon_path = args.bilingual_lexicon_path
flores_dir = args.flores_dir
kreyolmt_dir = args.kreyolmt_dir
madar_dir = args.madar_dir
batch_size = args.batch_size
prompting_strategy = args.prompting_strategy

mt_outputs_dir = f"{args.mt_outputs_dir}/{hrln}/{exp_key}"
results_dir = f"{args.results_dir}/{hrln}"

os.makedirs(mt_outputs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

device = torch.cuda.current_device() if torch.cuda.is_available() else -1
print(f"Device: {device}")

# configuring model
if model_name == "m2m" or model_name == "m2mL":
    iso3_to_m2mcode = {
        "hin": "hi",
        "tur": "tr",
        "arb": "ar",
        "rus": "ru",
        "ind": "id",
        "ita": "it",
        "hat": "ht",
        "jav_Latn": "jv",
        "sun_Latn": "su",
        "ceb_Latn": "ceb",
        "tgl_Latn": "tl",
        "ilo_Latn": "ilo",
        "zsm_Latn": "ms",
        "uzn_Latn": "uz",
        "fra_Latn": "fr",
        "por_Latn": "pt",
        "ita_Latn": "it",
        "ron_Latn": "ro",
        "glg_Latn": "gl",
        "cat_Latn": "ca",
        "oci_Latn": "oc",
        "ast_Latn": "ast",
        "spa_Latn": "es",
    }
    if model_path is None:
        model_path = "facebook/m2m100_1.2B"
    if lora:
        base_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = M2M100ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
    tokenizer.src_lang = iso3_to_m2mcode[crl] if crl in iso3_to_m2mcode else iso3_to_m2mcode[hrln]
    tokenizer.tgt_lang = "en"
    model.to(device)
elif model_name == "nllb":
    iso3_to_nllbcode = {
            "hin": "hin_Deva",
            "tur": "tur_Latn",
            "arb": "arb_Arab",
            "rus": "rus_Cyrl",
            "ind": "ind_Latn",
            "ita": "ita_Latn",
            "hat": "hat_Latn",
        }
    if "nllb_fft" in exp_key:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=iso3_to_nllbcode[hrln], tgt_lang="eng_Latn")
    else:
        base_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        peft_path = model_path
        model = PeftModel.from_pretrained(base_model, peft_path)
        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="hin_Deva", tgt_lang="eng_Latn")
    model.to(device)
elif model_name == "aya-23-8b":
    QUANTIZE_4BIT = True
    USE_GRAD_CHECKPOINTING = True
    TRAIN_MAX_SEQ_LENGTH = 512
    USE_FLASH_ATTENTION = False
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
        cache_dir="../models"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if model_path:
        model.load_adapter(model_path)
        
charbleu = False
print("Evaluating MT! ")
if charbleu:
    max_order = 18
else:
    max_order = 4
print(f"Char-level: {charbleu}, max order: {max_order}")

print(f"Evaluating...")

# Aya-23-8b specific functions
#----------------------------
def get_message_format(prompts):
  messages = []

  for p in prompts:
    messages.append(
        [{"role": "user", "content": p}]
      )

  return messages

def generate_aya_23(
      src_lang,
      inputs,
      model,
      temperature=0.3,
      top_p=0.75,
      top_k=0,
      max_new_tokens=40,
      prompting_strategy="nolangname"
    ):
    langname = ""
    hrln_name = ""
    if prompting_strategy == "nolangname":
        langname = ""
    elif prompting_strategy == "langname":
        langname = flores_code_to_langname(src_lang)
    elif prompting_strategy == "dialectofhrln":
        print(f"src_lang: {src_lang}")
        _, hrln_name = flores_code_to_hrln(src_lang)

    strategies = {
        "nolangname": "Translate into English: \n",
        "langname": f"Translate from {langname} into English: \n",
        "dialectofhrln": f"Translate from a dialect of {hrln_name} into English:\n"
    }  
  
    prompts = [strategies[prompting_strategy] + inp for inp in inputs]

    messages = get_message_format(prompts)

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
        )
    input_ids = input_ids.to(model.device)
    prompt_padded_len = len(input_ids[0])

    gen_tokens = model.generate(
        input_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        )

    # get only generated tokens
    gen_tokens = [
        gt[prompt_padded_len:] for gt in gen_tokens
    ]

    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return gen_text

#-----------------------------

def get_eval_dataset(hrln, lang, use_arb_madar = False):
    if "arb" in hrln and use_arb_madar: # Use MADAR test set

        CITY_CODE_TO_NAME = {
            "ale": "Aleppo",
            "alx": "Alexandria",
            "alg": "Algiers",
            "amm": "Amman",
            "asw": "Aswan",
            "bag": "Baghdad",
            "bas": "Basra",
            "bei": "Beirut",
            "ben": "Benghazi",
            "cai": "Cairo",
            "dam": "Damascus",
            "doh": "Doha",
            "fes": "Fes",
            "jed": "Jeddah",
            "jer": "Jerusalem",
            "kha": "Khartoum",
            "msa": "MSA",
            "mos": "Mosul",
            "mus": "Muscat",
            "rab": "Rabat",
            "riy": "Riyadh",
            "sal": "Salt",
            "san": "Sanaa",
            "sfx": "Sfax",
            "tri": "Tripoli",
            "tun": "Tunis",
            "eng": "English"
        }

        lang = CITY_CODE_TO_NAME[lang]
        inputs_file = f"{madar_dir}/MADAR.corpus.{lang}.tsv"
        references_file = f"{madar_dir}/MADAR.corpus.English.tsv"
        test_data = defaultdict(dict)
        with open(inputs_file, "r") as f:
            lines = f.readlines()
            lines = [line.strip().split("\t") for line in lines]
            for line in lines:
                if "test" not in line[1]:
                    continue
                sent_idx = line[0]
                source_sent = line[3]
                test_data[sent_idx]["source"] = source_sent

        with open(references_file, "r") as f:
            lines = f.readlines()
            lines = [line.strip().split("\t") for line in lines]
            for line in lines:
                sent_idx = line[0]
                if sent_idx not in test_data:
                    continue
                target_sent = line[3]
                test_data[sent_idx]["target"] = target_sent

        # Remove examples that don't have target
        test_data = {k: v for k, v in test_data.items() if "target" in v}
        test_data = list(test_data.values())
        test_data = [{"source": d["source"], "target": d["target"]} for d in test_data]
        dataset = Dataset.from_dict({"source": [d["source"] for d in test_data], "target": [d["target"] for d in test_data]})
    else:
        if "hat" in hrln:
            inputs_file = f"{kreyolmt_dir}/{lang}--eng/test.cleaned.{lang}"
            references_file = f"{kreyolmt_dir}/{lang}--eng/test.cleaned.eng"
        else:
            inputs_file = f"{flores_dir}/devtest/{lang}.devtest"
            references_file = f"{flores_dir}/devtest/eng_Latn.devtest"

        dataset = load_dataset("text", data_files={"source": [inputs_file], \
                "target": [references_file]})

        # Create a new dataset that has source and target as columns
        dataset = Dataset.from_dict({"source": dataset["source"]["text"], "target": dataset["target"]["text"]})
    return dataset
            
model.eval()
bleu_scores = dict()
lrl = crl.split("_")[0]

use_arb_madar = madar_dir is not None and "arb" in hrln
if "arb" in hrln and use_arb_madar:
    CITY_TO_LANG = {
        "bag": "acm_Arab",
        "cai": "arz_Arab",
        "dam": "apc_Arab",
        "fes": "ary_Arab",
        "jer": "ajp_Arab",
        "riy": "ars_Arab",
        "san": "acq_Arab",
        "tun": "aeb_Arab"
    }
    lrl = CITY_TO_LANG[crl].split("_")[0]
else:
    lrl = crl.split("_")[0]

lang_pair = lrl + '-' + hrln

print(f"Evaluating for: {crl}")

dataset = get_eval_dataset(hrln, crl, use_arb_madar=use_arb_madar)
    
input_sents = dataset["source"]
true_sents = dataset["target"]
pred_sents = list()

bilingual_lexicon_path = f"{bilingual_lexicon_path}/{lang_pair}/{lang_pair}_{denoise_func}.json"
denoiser = Denoiser(lrl, hrln, bilingual_lexicon_path)
# denoise dataset based on parameter
if denoise_func == "functional":
    dataset = dataset.map(lambda x: {"source":x["source"], "denoised_source": denoiser.denoise_all(x["source"]), "target": x["target"]})
elif denoise_func == "content":
    dataset = dataset.map(lambda x: {"source":x["source"], "denoised_source": denoiser.denoise_all(x["source"]), "target": x["target"]})
elif denoise_func == "all":
    dataset = dataset.map(lambda x: {"source":x["source"], "denoised_source": denoiser.denoise_all(x["source"]), "target": x["target"]})
else:
    raise ValueError(f"Invalid denoise_func value: '{denoise_func}'. Expected one of 'functional', 'content', or 'all'.")
    
original_input_sents = dataset["source"]
input_sents = dataset["denoised_source"]
true_sents = dataset["target"]

changed_words = sum([denoiser.report_changed_words(input, denoised_input) for input, denoised_input in zip(original_input_sents, input_sents)])
total_words = sum([len(input.split()) for input in original_input_sents])
cumulative_changed_words_ratio = changed_words/total_words
print(f"Changed words: {changed_words}, Total words: {total_words}, Changed words ratio: {cumulative_changed_words_ratio}")
    
for batch in tqdm(dataset.iter(batch_size=batch_size)):
    inputs = batch["denoised_source"]

    if model_name in {"m2m", "m2mL", "nllb"}:
        tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
        # For M2M:
        if "m2m" in exp_key:
            generated_tokens = model.generate(**tokenized_inputs, forced_bos_token_id=tokenizer.get_lang_id("en"))
        # For NLLB:
        else:
            generated_tokens = model.generate(**tokenized_inputs)
        pred = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_sents.extend(pred)
    
    elif model_name in {"aya-23-8b"}:
        inputs = batch["denoised_source"]
        pred = generate_aya_23(crl, inputs, model, prompting_strategy=prompting_strategy)
        pred = [p.split("\n")[0] for p in pred]
        pred_sents.extend(pred)

print(f"True sentences: {len(true_sents)}")
print(f"Predicted sentences: {len(pred_sents)}")
print(f"Sample true sentence: {true_sents[:3]}")
print(f"Sample predicted sentence: {pred_sents[:3]}")

output_path = os.path.join(mt_outputs_dir, f"{crl[:3]}-eng.json") #if not charbleu else os.path.join(output_dir, crl+"_preds_charbleu.txt")

if charbleu:
    # Put space after each character
    pred_sents = [" ".join(list(pred_sent)) for pred_sent in pred_sents]
    true_sents = [[" ".join(list(true_sent[0]))] for true_sent in true_sents]

# Find BLEU score
bleu = evaluate.load("bleu")
metric = bleu.compute(predictions=pred_sents, references=true_sents, max_order=max_order)
score = round(metric["bleu"]*100,2)
bleu_scores[crl] = score
print(f"BLEU scores for {crl}: {bleu_scores}")

outputs = []
for original_input, modified_input, output, reference in zip(original_input_sents, input_sents, pred_sents, true_sents):
    changed_word_ratio = denoiser.report_changed_words(original_input, modified_input) / len(original_input.split())
    try:
        bleu_score = round(bleu.compute(
            predictions=[output], 
            references=[[reference]],
            max_order=max_order
        )["bleu"], 2)
    except ZeroDivisionError:
        bleu_score = 0.0
    output_dict = {
        "raw_input": original_input,
        "denoised_input": modified_input,
        "output": output,
        "reference": reference,
        "changed_word_ratio": round(changed_word_ratio, 2),
        "bleu": bleu_score
    }
    outputs.append(output_dict)

with open(output_path, "w") as f:
    json.dump(outputs, f, indent = 2, ensure_ascii=False)

# Writing results to a file
## If the file already exists, load the results and update the results
results_path = f"{results_dir}/{exp_key}.json"
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)
        results[crl] = {"bleu": score, "changed_word_ratio":cumulative_changed_words_ratio}
else:
    results = {}
    results["hrln"] = hrln
    results["exp_key"] = exp_key
    results["model_path"] = model_path
    results["dictionary_source"] = bilingual_lexicon_path
    results[crl] = {"bleu": score, "changed_word_ratio":cumulative_changed_words_ratio}

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)