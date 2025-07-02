from .noise import Noise
from .phonological import PhonologicalNoiserAugmenter
import random
from collections import defaultdict, Counter
import numpy as np
import json
import sys
import os

from .utils.misc import normalize_lang_codes, get_character_set

# sys.path.append(os.getcwd())

# from scipy.stats import chisquare

random.seed(42)
np.random.seed(42)

class RandomWordNoiserAugmenter(Noise):
    '''
    Noise type: switch out words from the input with non-words
    Required params: text_file, theta_global
    Input format: {param: value}
    '''
    
    def __init__(self, noise_params):
        '''Initialize noise with noise parameters
        Args:
            noise_params: dict, noise parameters, like {theta_1: 0.5}
            Should contain:
                lang: str, language code
                text_file: str, text file to initialize vocabulary
                theta_random_word: float, probability of switching out a word
            Also accepts:
                output_dir: str, output directory
        '''
        self.class_name = "RandomWordNoiserAugmenter"
        self.required_keys = {"lang", "text_file", "theta_random_word"}
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
        self.vocab = self.get_vocab(self.text_file)
        
        if hasattr(self, "output_dir"):
            os.makedirs(self.output_dir, exist_ok=True)
        

    def get_vocab(self, text_file):
        '''Initialize vocabulary from vocab file'''
        print(f"Initializing vocabulary from {text_file}...")
        vocab = defaultdict(lambda: 0)
        for line in open(text_file):
            words = line.strip().split()
            for word in words:
                # Remove punctuation
                punctuation_and_bad_chars = "»«.,!?()[]{}\"'`:;'/\\-–—~_<>|@#$%^&*+=\u200b\u200c\u200d\u200e\u200f।"
                word = word.strip(punctuation_and_bad_chars)
                # If word has numeric characters, skip
                if any(char.isdigit() for char in word):
                    continue
                # All characters in word must be in character set
                if not all(char in self.character_set for char in word):
                    # print(f"Not in character set: {word}")
                    continue
                                
                vocab[word.lower()] += 1
        print(f"Finished initializing vocabulary from {text_file}!")
        print(f"Length of vocab: {len(vocab)}")

        return vocab
    
    def apply_noise(self, input):
        '''Apply noise to input
        Args:
            input: str, input text
        Returns:
            str, noised text
        '''
        # Apply noise
        noised_input = list()

        for idx, input_word in enumerate(input.split()):
            if input_word[0].isupper() and idx != 0:
                # We do not affect proper nouns
                noised_input.append(input_word)
                continue

            # If word has numeric characters, skip
            if any(char.isdigit() for char in input_word):
                noised_input.append(input_word)
                continue

            # All characters in word must be in character set
            punctuation_and_bad_chars = "।»«.,!?()[]{}\"'`:;'/\\-–—~_<>|@#$%^&*+=\u200b\u200c\u200d\u200e\u200f"

            if random.random() < self.theta_random_word:
                # Switch out the word for a different word
                candidates = set(self.vocab.keys()) - {input_word.lower()}
                noised_word = random.choice(list(candidates))
                # If there was punctuation, add it back
                if input_word[-1] in punctuation_and_bad_chars:
                    noised_word += input_word[-1]
                noised_input.append(noised_word)
            else:
                noised_input.append(input_word)

        return " ".join(noised_input)
