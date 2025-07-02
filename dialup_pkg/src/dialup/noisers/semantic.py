from .noise import Noise
# from phonological import PhonologicalNoiserAugmenter
import random
from collections import defaultdict, Counter
import numpy as np
import json
import sys
import os

from .utils.misc import normalize_lang_codes, get_character_set

# sys.path.append(os.getcwd())


from .utils.get_functional_words import get_tag2wordlist
from .utils.get_synonyms import get_synonyms

# from scipy.stats import chisquare

random.seed(42)
np.random.seed(42)


class SemanticNoiserAugmenter(Noise):
    '''
    Noise type: switch out words from the vocabulary with synonyms, with the correct inflection.
    Only affects content words.
    Required params: text_file, theta_semantic_global
    Input format: {param: value}
    '''
    
    def __init__(self, noise_params):
        '''Initialize noise with noise parameters
        Args:
            noise_params: dict, noise parameters, like {theta_1: 0.5}
            Should contain:
                lang: str, language code
                text_file: str, txt file
                theta_semantic_global: float, probability of switching out a content word with a non-word
            Also accepts:
                output_dir: str, output directory
        '''
        import stanza
        import snowballstemmer

        self.class_name = "SemanticNoiserAugmenter"
        self.required_keys = {"lang", "text_file", "theta_semantic_global"}
        self.allowed_keys = {"output_dir"}
        self.check_noise_params(noise_params)

        for key in noise_params:
            if key == "lang":
                self.lang = normalize_lang_codes(noise_params[key])
                continue
            setattr(self, key, noise_params[key])

        _, self.character_set = get_character_set(self.lang)

        print(f"Character set: {self.character_set}")

        # Initialize vocabulary
        # self.vocab = self.get_vocab(self.text_file)
        self.tag2wordlist = get_tag2wordlist(self.lang)
        # self.vocab_map = self.construct_new_vocab()

        # Initialize lemmatizer and stemmer
        lang2stanza_code = {
            "hin": "hi",
            "tur": "tr",
            "ita": "it",
            "hat": "ht",
            "ind": "id",
            "arb": "ar",
        }
        self.nlp = stanza.Pipeline(lang2stanza_code[self.lang], processors='tokenize,pos,lemma')

        lang2stemmer_code = {
            "hin": "hindi",
            "tur": "turkish",
            "ita": "italian",
            "hat": "english",
            "ind": "indonesian",
            "arb": "arabic",
        }
        self.stemmer = snowballstemmer.stemmer(lang2stemmer_code[self.lang])

        # Paths to precomputed lemmas and stems
        # Get absolute path of utils directory
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        self.stems_path = f"{utils_dir}/utils/lemmas_and_stems/stems/{self.lang}.json"
        self.lemmas_path = f"{utils_dir}/utils/lemmas_and_stems/lemmas/{self.lang}.json"
        with open(self.stems_path) as f:
            self.word2stem = json.load(f)
        with open(self.lemmas_path) as f:
            self.word2lemma = json.load(f)

        # Initialize synonym finder function
        if self.lang == "hin":
            from pyiwn import iwn
            iwnobj = iwn.IndoWordNet(iwn.Language.HINDI)
            self.get_synonyms = lambda word: get_synonyms(word, self.lang, hin_wn=iwnobj)
        
        elif self.lang == "tur":
            # https://github.com/StarlangSoftware/TurkishWordNet-Py
            from WordNet.WordNet import WordNet 
            wordnet = WordNet()
            self.get_synonyms = lambda word: get_synonyms(word, self.lang, tur_wn=wordnet)
        
        elif self.lang in {"arb", "ita", "ind"}:
            import wn
            wn_key = {
                "arb": "omw-arb:1.4",
                "ita": "omw-iwn:1.4",
                "ind": "omw-id:1.4",
            }
            wn.download(wn_key[self.lang])
            self.get_synonyms = lambda word: get_synonyms(word, self.lang, wn=wn, wn_key=wn_key[self.lang])
        
        else:
            raise NotImplementedError(f"Language {self.lang} not supported yet.")

        if hasattr(self, "output_dir"):
            os.makedirs(self.output_dir, exist_ok=True)

        
    def is_word_functional(self, word):
        '''Check if word is functional'''
        for tag in self.tag2wordlist:
            if word in self.tag2wordlist[tag]:
                return True
        return False

    def get_vocab_map_stats(self):
        '''
        Get stats of vocab map
        '''
        self.get_vocab = self.get_vocab(self.text_file)
        stats = dict()
        stats["theta_semantic_global"] = self.theta_content_global
        stats["vocab_size"] = len(self.vocab)
        stats["content_words"] = len([word for word in self.vocab if not self.is_word_functional(word)])
        stats["func_words"] = len([word for word in self.vocab if self.is_word_functional(word)])
        stats["content_words_noised_frac"] = round(len([word for word in self.vocab_map if not self.is_word_functional(word) \
                                                 and self.vocab_map[word] != word]) / stats["content_words"], 2)
        
        return stats
        
    def noise_word(self, word):
        '''
        This function switches out a word with a synonym, with the correct inflection.
        '''
        # Lemmatize the input word
        if word in self.word2lemma:
            lemma = self.word2lemma[word]
        else:
            # print(f"Computing lemma from scratch")
            doc = self.nlp(word)
            try:
                lemma = doc.sentences[0].words[0].lemma
            except IndexError:
                return word
        # Get synonyms of the lemma
        syns = self.get_synonyms(lemma)
        if len(syns) == 0: # No synonyms found
            return word
        # Randomly choose a synonym
        noised_lemma = random.choice(syns)
        # Get inflections of the input word
        input_word_stem = self.stemmer.stemWord(word) if word in self.word2stem else self.stemmer.stemWord(lemma)
        input_word_inflection = word[len(input_word_stem):]

        # Stem the new word and add the inflection of the input word to the synonym
        noised_stem = self.stemmer.stemWord(noised_lemma) if noised_lemma in self.word2stem else noised_lemma
        noised_word = noised_stem + input_word_inflection

        return noised_word

    def construct_new_vocab(self, input):
        '''
        We construct a new vocabulary map based on the input text.
        With probability theta_semantic_global, switch out a word from the vocabulary.
        If it's a content word, we'll change it to a synonym, and then inflect it correctly.
        Returns:
            vocab_map: dict, mapping of old word to new word
        '''
        vocab_map = dict()
        for word in input.split():
            word = word.lower()
            punctuation_and_bad_chars = "।»«.,!?()[]{}\"'`:;'/\\-–—~_<>|@#$%^&*+=\u200b\u200c\u200d\u200e\u200f"
            word = word.strip(punctuation_and_bad_chars)
            # If word has numeric characters, skip
            if any(char.isdigit() for char in word):
                vocab_map[word] = word
                continue
            # All characters in word must be in character set
            if not all(char in self.character_set for char in word):
                # print(f"Not in character set: {word}")
                vocab_map[word] = word
                continue
            # Word must not be empty
            if len(word) == 0:
                vocab_map[word] = word
                continue
            
            # If word is functional, we don't change it
            if self.is_word_functional(word):
                continue
            # If word is content:
            if random.random() < self.theta_semantic_global:
                # print(f"Switching out {word}")
                new_word = self.noise_word(word)
                vocab_map[word] = new_word
            else:
                vocab_map[word] = word
        
        return vocab_map


    def apply_noise(self, input):
        '''Apply noise to input
        Args:
            input: str, input text
        Returns:
            str, noised text
        '''
        
        vocab_map = self.construct_new_vocab(input)

        # Apply noise
        noised_input = list()

        for idx, input_word in enumerate(input.split()):
            if input_word[0].isupper() and idx != 0:
                # We do not affect proper nouns
                noised_input.append(input_word)
                continue
            word = input_word.strip(".,!?।").lower()
            if word in vocab_map:
                mapped_word = vocab_map[word]
                # Capitalize first letter if original word was capitalized
                if input_word[0].isupper():
                    mapped_word = mapped_word.capitalize()
                # Add punctuation back
                if input_word[-1] in ".,!?।":
                    mapped_word += input_word[-1]
                noised_input.append(mapped_word)
            else:
                noised_input.append(input_word)
        return " ".join(noised_input)


