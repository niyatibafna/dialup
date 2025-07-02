'''
This code is to generate wordlists for closed class tags in UD data, 
and save these as JSON files in ud_closed_class_wordlists/

This requires the ud-treebanks-v2.13 dataset to be downloaded and the file paths to be set in ud_data_filepaths.

This code takes in a conllu UD file and finds the most frequent tag for each word in the file.
For each word, it then checks if the tag is in the closed_class_tags list.
Finally, we create a JSON file with the wordlist per closed_class tag.
'''

import os, sys
from collections import defaultdict
import json
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .get_functional_words import ud_wordlists_paths, closed_class_tags, OUTDIR

ud_data_filepaths = {
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

def read_conllu(filename):
    import conllu
    conllu_file = open(filename, "r", encoding="utf-8")
    conllu_data = conllu_file.read()
    conllu_sentences = conllu.parse(conllu_data)
    return conllu_sentences

if __name__ == "__main__":
    for lang, file in ud_data_filepaths.items():
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
        
        os.makedirs(OUTDIR, exist_ok=True)
        print(f"Writing to: {ud_wordlists_paths[lang]}")
        
        with open(ud_wordlists_paths[lang], "w", encoding="utf-8") as out:
            json.dump(tag2wordlist, out, indent=2, ensure_ascii=False)
