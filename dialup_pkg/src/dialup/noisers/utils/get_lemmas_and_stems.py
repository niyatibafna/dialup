'''
We collect the vocabulary from text files, stem and lemmatize the words, and save the stems and lemmas in a dictionary.
'''

import os, sys
# sys.path.append("/export/b08/nbafna1/projects/generating_artificial_langs_for_dialectical_robustness/noisers/")
from .get_functional_words_from_ud import OUTDIR, closed_class_tags
from .get_functional_words_from_ud import output_paths as ud_wordlists_paths
from .get_synonyms import get_synonyms

import json
from collections import defaultdict

import stanza
import snowballstemmer
from tqdm import tqdm


lang = sys.argv[1]

# Collect vocabulary
print(f"Collecting vocabulary for {lang}...")

read_files = {
    "ita": "/export/b08/nbafna1/data/wikimatrix/en-it/WikiMatrix.en-it.it",
    "arb": "/export/b08/nbafna1/data/wikimatrix/ar-en/WikiMatrix.ar-en.ar",
    "ind": "/export/b08/nbafna1/data/wikimatrix/en-id/WikiMatrix.en-id.id",
    "tur": "/export/b08/nbafna1/data/wikimatrix/en-tr/WikiMatrix.en-tr.tr",
    "hin": "/export/b08/nbafna1/data/wikimatrix/en-hi/WikiMatrix.en-hi.hi",
}

def get_vocab(text_file):
    vocab = defaultdict(lambda: 0)
    for line in open(text_file):
        words = line.strip().split()
        for word in words:
            # Remove punctuation
            punctuation_and_bad_chars = "।»«.,!?()[]{}\"'`:;'/\\-–—~_<>|@#$%^&*+=\u200b\u200c\u200d\u200e\u200f"
            word = word.strip(punctuation_and_bad_chars)
            # If word has numeric characters, skip
            if any(char.isdigit() for char in word):
                continue
                
            vocab[word.lower()] += 1

    print(f"Finished initializing vocabulary from {text_file}!")
    print(f"Length of vocab: {len(vocab)}")
    return vocab


vocab = get_vocab(read_files[lang])


# Init lemmatizer and stemmer
lang2stanza_code = {
    "hin": "hi",
    "spa": "es",
    "tur": "tr",
    "ita": "it",
    "hat": "ht",
    "ind": "id",
    "arb": "ar",
}
nlp = stanza.Pipeline(lang2stanza_code[lang], processors='tokenize,pos,lemma')

lang2stemmer_code = {
    "hin": "hindi",
    "spa": "spanish",
    "tur": "turkish",
    "ita": "italian",
    "hat": "english",
    "ind": "indonesian",
    "arb": "arabic",
}
stemmer = snowballstemmer.stemmer(lang2stemmer_code[lang])


# Collect lemmas and stems
print(f"Collecting lemmas and stems for {lang}...")
word2stem = {}
word2lemma = {}                                         

for word in tqdm(vocab.keys()):
    try:
        doc = nlp(word)
        lemma = doc.sentences[0].words[0].lemma
    except IndexError:
        lemma = word
    stem = stemmer.stemWord(lemma)
    word2stem[word] = stem
    word2lemma[word] = lemma


stem_outfile = f"/export/b08/nbafna1/projects/generating_artificial_langs_for_dialectical_robustness/noisers/utils/lemmas_and_stems/stems/{lang}.json"
with open(stem_outfile, "w") as f:
    json.dump(word2stem, f, indent=4, ensure_ascii=False)

lemma_outfile = f"/export/b08/nbafna1/projects/generating_artificial_langs_for_dialectical_robustness/noisers/utils/lemmas_and_stems/lemmas/{lang}.json"
with open(lemma_outfile, "w") as f:
    json.dump(word2lemma, f, indent=4, ensure_ascii=False)



