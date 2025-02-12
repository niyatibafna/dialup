import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, NllbTokenizer, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, DataCollatorForSeq2Seq, BitsAndBytesConfig, HfArgumentParser
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import pipeline
from peft import PeftModel 
import evaluate
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os, sys
import json
from argparse import ArgumentParser
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
from trl import SFTTrainer
import bitsandbytes as bnb

from collections import defaultdict

sys.path.append("/export/b08/nbafna1/projects/dialectical-robustness-mt/utils/")
from flores_code_to_langname import flores_code_to_langname, get_crls, flores_code_to_hrln


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--exp_key", type=str, required=True, help="Experiment key")
    parser.add_argument("--hrln", type=str, required=True, help="High-resource language name")
    parser.add_argument("--crl", type=str, required=True, help="CRL language name")
    parser.add_argument("--model_name",type=str, required=True, help="Model to finetune, must be one of 'm2m', 'nllb'")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--lora", action="store_true", help="Use LoRA model")
    parser.add_argument("--flores_dir", type=str, required=True, help="Path to FloRes data")
    parser.add_argument("--kreyolmt_dir", type=str, required=True, help="Path to KreyolMT data, for evaluating on Haitian")
    parser.add_argument("--madar_dir", type=str, required=False, help="Path to MADAR data, for evaluating on Arabic dialects")
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
model_path = args.model_path
lora = args.lora
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
    # tokenizer.src_lang = iso3_to_m2mcode[hrln]
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
# Get FloRes source sentences

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
    # print(f"Example prompt: {prompts[0]}")

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
        # dataset = dataset.select(range(300))
        
    else: 
        if "hat" in hrln:
            inputs_file = f"{kreyolmt_dir}/{lang}--eng/test.cleaned.{lang}"
            references_file = f"{kreyolmt_dir}/{lang}--eng/test.cleaned.eng"

        else: # Use FloRes test set
            inputs_file = f"{flores_dir}/devtest/{lang}.devtest"
            references_file = f"{flores_dir}/devtest/eng_Latn.devtest"

        dataset = load_dataset("text", data_files={"source": [inputs_file], \
                "target": [references_file]})

        dataset = Dataset.from_dict({"source": dataset["source"]["text"], "target": dataset["target"]["text"]})

    return dataset


model.eval()
bleu_scores = dict()
# comet_scores = dict()
lang = crl

print(f"Evaluating for: {lang}")

dataset = get_eval_dataset(hrln, lang)

input_sents = dataset["source"]
true_sents = dataset["target"]
pred_sents = list()
for batch in tqdm(dataset.iter(batch_size=batch_size)):
    inputs = batch["source"]

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
        inputs = batch["source"]
        pred = generate_aya_23(lang, inputs, model, prompting_strategy=prompting_strategy)
        pred = [p.split("\n")[0] for p in pred]
        pred_sents.extend(pred)

print(f"True sentences: {len(true_sents)}")
print(f"Predicted sentences: {len(pred_sents)}")
print(f"Sample true sentence: {true_sents[:3]}")
print(f"Sample predicted sentence: {pred_sents[:3]}")

output_path = os.path.join(mt_outputs_dir, f"preds_{lang[:3]}-eng.json") #if not charbleu else os.path.join(output_dir, lang+"_preds_charbleu.txt")

if charbleu:
    # Put space after each character
    pred_sents = [" ".join(list(pred_sent)) for pred_sent in pred_sents]
    true_sents = [[" ".join(list(true_sent[0]))] for true_sent in true_sents]


# Find BLEU score
bleu = evaluate.load("bleu")
metric = bleu.compute(predictions=pred_sents, references=true_sents, max_order=max_order)
score = metric["bleu"]
bleu_scores[lang] = score
print(f"BLEU scores for {lang}: {bleu_scores}")

# Uncomment to find COMET score
# comet_metric = evaluate.load("comet")
# comet_score = comet_metric.compute(predictions=pred_sents, references=true_sents, sources=input_sents)
# comet_score = comet_score["mean_score"]
# comet_scores[lang] = comet_score
# print(f"COMET score for {lang}: {comet_score}")


outputs = [{"input": input, "output": output, "reference": reference} for input, output, reference in zip(input_sents, pred_sents, true_sents)]
with open(output_path, "w") as f:
    json.dump(outputs, f, indent = 2, ensure_ascii=False)


# Writing results to a file
## If the file already exists, load the results and update the results

results_file = f"{results_dir}/res-{exp_key}.json"

if os.path.exists(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
        results["bleu"][lang] = score
        # results["comet"][lang] = comet_score
else:
    results = {}
    results["hrln"] = hrln
    results["exp_key"] = exp_key
    results["bleu"] = bleu_scores
    # results["comet"] = comet_scores
    results["model_path"] = model_path


with open(f"{results_dir}/res-{exp_key}.json", "w") as f:
    json.dump(results, f, indent=2)