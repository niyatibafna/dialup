'''
This code takes in a conllu UD file and finds the most frequent tag for each word in the file.
For each word, it then checks if the tag is in the closed_class_tags list.
Finally, we create a JSON file with the wordlist per closed_class tag.
'''

import sys
from collections import defaultdict
import json

files = {
    "deu":"/export/b08/nbafna1/data/ud-treebanks-v2.13/UD_German-HDT/de_hdt-ud-train.conllu",
    "hin": "/export/b08/nbafna1/data/ud-treebanks-v2.13/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu",
    "arb": "/export/b08/nbafna1/data/ud-treebanks-v2.13/UD_Arabic-PADT/ar_padt-ud-train.conllu",
    "ind": "/export/b08/nbafna1/data/ud-treebanks-v2.13/UD_Indonesian-GSD/id_gsd-ud-train.conllu",
    "eng": "/export/b08/nbafna1/data/ud-treebanks-v2.13/UD_English-EWT/en_ewt-ud-train.conllu",
    "rus": "/export/b08/nbafna1/data/ud-treebanks-v2.13/UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu",
    "spa": "/export/b08/nbafna1/data/ud-treebanks-v2.13/UD_Spanish-GSD/es_gsd-ud-train.conllu",
    "fra": "/export/b08/nbafna1/data/ud-treebanks-v2.13/UD_French-GSD/fr_gsd-ud-train.conllu",
    "tur": "/export/b08/nbafna1/data/ud-treebanks-v2.13/UD_Turkish-IMST/tr_imst-ud-train.conllu",
    "ita": "/export/b08/nbafna1/data/ud-treebanks-v2.13/UD_Italian-ISDT/it_isdt-ud-train.conllu",

}

OUTDIR = "/export/b08/nbafna1/projects/generating_artificial_langs_for_dialectical_robustness/noisers/utils/ud_closed_class_wordlists"
output_paths ={
    "hin": f"{OUTDIR}/hi_hdtb-ud-train.json",
    "deu": f"{OUTDIR}/de_hdt-ud-train.json",
    "arb": f"{OUTDIR}/ar_padt-ud-train.json",
    "ind": f"{OUTDIR}/id_gsd-ud-train.json",
    "eng": f"{OUTDIR}/en_ewt-ud-train.json",
    "rus": f"{OUTDIR}/ru_syntagrus-ud-train.json",
    "spa": f"{OUTDIR}/es_gsd-ud-train.json",
    "fra": f"{OUTDIR}/fr_gsd-ud-train.json",
    "tur": f"{OUTDIR}/tr_imst-ud-train.json",
    "ita": f"{OUTDIR}/it_isdt-ud-train.json",
}

closed_class_tags = ['ADP', 'AUX', 'CCONJ', 'DET', 'PART', 'PRON', 'SCONJ']


def read_conllu(filename):
    import conllu
    conllu_file = open(filename, "r", encoding="utf-8")
    conllu_data = conllu_file.read()
    conllu_sentences = conllu.parse(conllu_data)
    return conllu_sentences

if __name__ == "__main__":
    for lang, file in files.items():
        if lang not in {"tur", "ita"}:
            continue
        print(f"Processing: {lang}...")
        word2tags = defaultdict(lambda: defaultdict(lambda: 0))
        sentences = read_conllu(file)
        for sentence in sentences:
            for token in sentence:
                word = token["form"].lower()
                tag = token["upos"]
                word2tags[word][tag] += 1

        tag2wordlist = defaultdict(lambda: [])
        for word in word2tags:
            tag = max(word2tags[word], key=word2tags[word].get)
            if tag in closed_class_tags:
                tag2wordlist[tag].append(word)
        
        print(f"Writing to: {output_paths[lang]}")
        
        with open(output_paths[lang], "w", encoding="utf-8") as out:
            json.dump(tag2wordlist, out, indent=2, ensure_ascii=False)