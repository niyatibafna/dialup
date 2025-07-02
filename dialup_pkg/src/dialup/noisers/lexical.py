from .noise import Noise
from .phonological import PhonologicalNoiserAugmenter, GlobalPhonologicalNoiser
import random
from collections import defaultdict, Counter
import numpy as np
import json
import sys
import os

from .utils.misc import normalize_lang_codes, get_character_set

# sys.path.append(os.getcwd())

from .utils.get_functional_words import get_tag2wordlist

from scipy.stats import chisquare

random.seed(42)
np.random.seed(42)

class LexicalNoiserAugmenter(Noise):
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
                text_file: str, txt file
                theta_content_global: float, probability of switching out a content word with a non-word
                theta_func_global: float, probability of switching out a function word with a non-word
            Also accepts:
                chargram_length: int, character n-gram length
                output_dir: str, output directory
        '''
        self.class_name = "LexicalNoiserAugmenter"
        self.required_keys = {"lang", "text_file", "theta_content_global", "theta_func_global"}
        self.allowed_keys = {"chargram_length", "output_dir"}
        self.check_noise_params(noise_params)

        for key in noise_params:
            if key == "lang":
                self.lang = normalize_lang_codes(noise_params[key])
                continue
            setattr(self, key, noise_params[key])

        if not hasattr(self, "chargram_length"):
            self.chargram_length = 3
        
        _, self.character_set = get_character_set(self.lang)

        print(f"Character set: {self.character_set}")

        # We'll use a phonological noiser for function words
        self.phon_noiser = PhonologicalNoiserAugmenter({"lang": self.lang, "theta_phon": 0.5, "text_file": self.text_file})

        # Initialize vocabulary
        self.vocab = self.get_vocab(self.text_file)
        self.chargram_models = self.train_chargram_model(self.chargram_length)
        self.tag2wordlist = get_tag2wordlist(self.lang)

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
                punctuation_and_bad_chars = "»«.,!?()[]{}\"'`:;'/\\-–—~_<>|@#$%^&*+=\u200b\u200c\u200d\u200e\u200f"
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


    def train_chargram_model(self, chargram_length=3):
        '''Train a character n-gram model on text
        Args:
            n: int, char n-gram length
        Returns:
            chargram_models: dict, contains character n-gram models for all n <= chargram_length {n: model} 
                - model: defaultdict, character n-gram model of type {prefix: {suffix: count}}
        '''
        print(f"Training chargram model with chargram length {chargram_length}...")
        chargram_models = dict() # contains chargram_models for all cgram lengths <= chargram_length
        for n in range(1, chargram_length + 1):
            chargram_model = defaultdict(lambda: defaultdict(lambda: 0)) # {prefix: {suffix: count}}

            for word in self.vocab:
                word = "!" + word # Add start token
                for i in range(len(word) - n + 1):
                    ngram = word[i:i+n]
                    # Increment count for ngram
                    chargram_model[ngram[:-1]][ngram[-1]] += 1
        
            chargram_models[n] = chargram_model

        print(f"Finished training chargram model with chargram length {chargram_length}!")
        return chargram_models

    def generate_word(self, mean_length):
        '''
        This function is for generating a non-word using the character n-gram model. We will:
        1. Sample the length of the non-word from a Poisson centered around mean_length
        2. Use self.chargram_models to generate the rest of the non-word based on the length of prefix
        Args:
            mean_length: float, mean length of non-word
        '''

        # Sample length of non-word from Poisson, must be at least 1

        length = max(1, np.random.poisson(mean_length))
        length += 1
        word = "!"

        while word == "!" or word[1:].lower() in self.vocab:
            # If generated word in vocab, generate another word
            word = "!"
            for _ in range(length):
                if len(word) < self.chargram_length-1:
                    # If the word is shorter than the length of the chargram model, 
                    # we will apply chargram model of the length of the word
                    chargram_model = self.chargram_models[len(word) + 1]
                    prefix = word
                else:
                    # If the word is longer than the prefix of the chargram model, 
                    # we will apply the chargram model of the length of the model
                    chargram_model = self.chargram_models[self.chargram_length]
                    prefix = word[-(self.chargram_length-1):]
                while len(list(chargram_model[prefix].keys())) == 0:
                    # Backing off to shorter chargram models
                    chargram_model = self.chargram_models[len(prefix)]
                    prefix = prefix[1:]
                # print(list(chargram_model[prefix].keys()))
                # print(len(list(chargram_model[prefix].keys())))

                # Sample the next character based on the prefix
                try:
                    next_char = np.random.choice(list(chargram_model[prefix].keys()), p=np.array(list(chargram_model[prefix].values())) / sum(chargram_model[prefix].values()))
                except:
                    print(f"Error: {prefix} not in chargram model")
                    # print(f"Chargram model: {chargram_model}")
                    print(f"Word: {word}")
                    print(f"Length: {length}")
                    print(f"Mean Length: {mean_length}")
                    print(f"Chargram Length: {self.chargram_length}")
                    print(f"Prefix: {prefix}")
                    print(f"{not chargram_model[prefix]}")
                    print(f"Chargram model: {chargram_model[prefix].keys()}")

                    raise
                word += next_char

        return word[1:]
    

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
        stats = dict()
        stats["theta_content_global"] = self.theta_content_global
        stats["theta_func_global"] = self.theta_func_global
        stats["vocab_size"] = len(self.vocab)
        stats["content_words"] = len([word for word in self.vocab if not self.is_word_functional(word)])
        stats["func_words"] = len([word for word in self.vocab if self.is_word_functional(word)])
        stats["content_words_noised_frac"] = round(len([word for word in self.vocab_map if not self.is_word_functional(word) \
                                                 and self.vocab_map[word] != word]) / stats["content_words"], 2)
        stats["func_words_noised_frac"] = round(len([word for word in self.vocab_map if self.is_word_functional(word)\
                                               and self.vocab_map[word] != word])/stats["func_words"], 2)
        return stats
        
    def construct_new_vocab(self, input):
        '''
        We construct a new vocabulary map based on the input text.
        With probability theta_global_*, switch out a word from the vocabulary.
        If it's a functional word, we'll apply phonological noise to it.
        If it's a content word, we'll change it to a non-word.
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
            
            # If word is functional:
            if self.is_word_functional(word):
                if random.random() < self.theta_func_global:
                    # print(f"Switching out {word}")
                    new_word = self.phon_noiser.apply_noise_for_sure(word)
                    vocab_map[word] = new_word
                else:
                    vocab_map[word] = word
                continue
            # If word is content:
            if random.random() < self.theta_content_global:
                # print(f"Switching out {word}")
                new_word = self.generate_word(len(word))
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
                # We do not affect proper nouns. This is a simple heuristic.
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



class GlobalLexicalNoiser(Noise):
    '''
    Noise type: switch out words from the vocabulary with non-words, and apply this change globally to every occurrence.
    Required params: text_file, theta_global
    Input format: {param: value}
    '''
    
    def __init__(self, noise_params):
        '''Initialize noise with noise parameters
        Args:
            noise_params: dict, noise parameters, like {theta_1: 0.5}
            Should contain:
                text_file: str, txt file
                theta_content_global: float, probability of switching out a content word with a non-word
                theta_func_global: float, probability of switching out a function word with a non-word
            Also accepts:
                chargram_length: int, character n-gram length
                output_dir: str, output directory
        '''
        self.class_name = "GlobalLexicalNoiser"
        self.required_keys = {"lang", "text_file", "theta_content_global", "theta_func_global"}
        self.allowed_keys = {"chargram_length", "output_dir"}
        self.check_noise_params(noise_params)

        for key in noise_params:
            if key == "lang":
                self.lang = normalize_lang_codes(noise_params[key])
                continue
            setattr(self, key, noise_params[key])

        if not hasattr(self, "chargram_length"):
            self.chargram_length = 3
        
        _, self.character_set = get_character_set(self.lang)

        print(f"Character set: {self.character_set}")

        # We'll use a phonological noiser for function words
        self.phon_noiser = GlobalPhonologicalNoiser({"lang": self.lang, "theta_phon": 0.5, "text_file": self.text_file})

        # Initialize vocabulary
        self.vocab = self.get_vocab(self.text_file)
        self.chargram_models = self.train_chargram_model(self.chargram_length)
        self.tag2wordlist = get_tag2wordlist(self.lang)
        self.vocab_map = self.construct_new_vocab()

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
                punctuation_and_bad_chars = "»«.,!?()[]{}\"'`:;'/\\-–—~_<>|@#$%^&*+=\u200b\u200c\u200d\u200e\u200f"
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

    def train_chargram_model(self, chargram_length=3):
        '''Train a character n-gram model on text
        Args:
            n: int, char n-gram length
        Returns:
            chargram_models: dict, contains character n-gram models for all n <= chargram_length {n: model} 
                - model: defaultdict, character n-gram model of type {prefix: {suffix: count}}
        '''
        print(f"Training chargram model with chargram length {chargram_length}...")
        chargram_models = dict() # contains chargram_models for all cgram lengths <= chargram_length
        for n in range(1, chargram_length + 1):
            chargram_model = defaultdict(lambda: defaultdict(lambda: 0)) # {prefix: {suffix: count}}

            for word in self.vocab:
                word = "!" + word # Add start token
                for i in range(len(word) - n + 1):
                    ngram = word[i:i+n]
                    # Increment count for ngram
                    chargram_model[ngram[:-1]][ngram[-1]] += 1
        
            chargram_models[n] = chargram_model

        print(f"Finished training chargram model with chargram length {chargram_length}!")
        return chargram_models

    def generate_word(self, mean_length):
        '''
        This function is for generating a non-word using the character n-gram model. We will:
        1. Sample the length of the non-word from a Poisson centered around mean_length
        2. Use self.chargram_models to generate the rest of the non-word based on the length of prefix
        Args:
            mean_length: float, mean length of non-word
        '''

        # Sample length of non-word from Poisson, must be at least 1

        length = max(1, np.random.poisson(mean_length))
        length += 1
        word = "!"

        while word == "!" or word[1:].lower() in self.vocab:
            # If generated word in vocab, generate another word
            word = "!"
            for _ in range(length):
                if len(word) < self.chargram_length-1:
                    # If the word is shorter than the length of the chargram model, 
                    # we will apply chargram model of the length of the word
                    chargram_model = self.chargram_models[len(word) + 1]
                    prefix = word
                else:
                    # If the word is longer than the prefix of the chargram model, 
                    # we will apply the chargram model of the length of the model
                    chargram_model = self.chargram_models[self.chargram_length]
                    prefix = word[-(self.chargram_length-1):]
                while len(list(chargram_model[prefix].keys())) == 0:
                    # Backing off to shorter chargram models
                    chargram_model = self.chargram_models[len(prefix)]
                    prefix = prefix[1:]
                # print(list(chargram_model[prefix].keys()))
                # print(len(list(chargram_model[prefix].keys())))

                # Sample the next character based on the prefix
                try:
                    next_char = np.random.choice(list(chargram_model[prefix].keys()), p=np.array(list(chargram_model[prefix].values())) / sum(chargram_model[prefix].values()))
                except:
                    print(f"Error: {prefix} not in chargram model")
                    # print(f"Chargram model: {chargram_model}")
                    print(f"Word: {word}")
                    print(f"Length: {length}")
                    print(f"Mean Length: {mean_length}")
                    print(f"Chargram Length: {self.chargram_length}")
                    print(f"Prefix: {prefix}")
                    print(f"{not chargram_model[prefix]}")
                    print(f"Chargram model: {chargram_model[prefix].keys()}")

                    raise
                word += next_char

        return word[1:]
    

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
        stats = dict()
        stats["theta_content_global"] = self.theta_content_global
        stats["theta_func_global"] = self.theta_func_global
        stats["vocab_size"] = len(self.vocab)
        stats["content_words"] = len([word for word in self.vocab if not self.is_word_functional(word)])
        stats["func_words"] = len([word for word in self.vocab if self.is_word_functional(word)])
        stats["content_words_noised_frac"] = round(len([word for word in self.vocab_map if not self.is_word_functional(word) \
                                                 and self.vocab_map[word] != word]) / stats["content_words"], 2)
        stats["func_words_noised_frac"] = round(len([word for word in self.vocab_map if self.is_word_functional(word)\
                                               and self.vocab_map[word] != word])/stats["func_words"], 2)
        return stats
        
    def construct_new_vocab_archive(self):
        '''
        With probability theta_global, switch out a word from the vocabulary with a non-word,
        applying appropriate theta for functional and content words
        Returns:
            vocab_map: dict, mapping of old word to new word
        '''
        vocab_map = dict()
        for word in self.vocab:

            if self.is_word_functional(word):
                theta = self.theta_func_global
            else:
                theta = self.theta_content_global

            if random.random() < theta:
                # print(f"Switching out {word}")
                new_word = self.generate_word(len(word))
                vocab_map[word] = new_word
            else:
                vocab_map[word] = word
        
        return vocab_map

    def construct_new_vocab(self):
        '''
        With probability theta_global_*, switch out a word from the vocabulary.
        If it's a functional word, we'll apply phonological noise to it.
        If it's a content word, we'll change it to a non-word.
        Returns:
            vocab_map: dict, mapping of old word to new word
        '''
        vocab_map = dict()
        for word in self.vocab:
            # If word is functional:
            if self.is_word_functional(word):
                if random.random() < self.theta_func_global:
                    # print(f"Switching out {word}")
                    new_word = self.phon_noiser.apply_noise_for_sure(word)
                    vocab_map[word] = new_word
                else:
                    vocab_map[word] = word
                continue
            # If word is content:
            if random.random() < self.theta_content_global:
                # print(f"Switching out {word}")
                new_word = self.generate_word(len(word))
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
        # Apply noise
        # For each word, map it to the corresponding word using the vocab map self.vocab_map

        noised_input = list()
        for input_word in input.split():
            if input_word[0].isupper():
                # We do not affect proper nouns
                noised_input.append(input_word)
                continue
            word = input_word.strip(".,!?").lower()
            if word in self.vocab_map:
                mapped_word = self.vocab_map[word]
                # Capitalize first letter if original word was capitalized
                if input_word[0].isupper():
                    mapped_word = mapped_word.capitalize()
                # Add punctuation back
                if input_word[-1] in ".,!?":
                    mapped_word += input_word[-1]
                noised_input.append(mapped_word)
            else:
                noised_input.append(input_word)
        return " ".join(noised_input)

    def record_noiser_artifacts(self):
        '''Record vocab map, number of words switched out'''
        if hasattr(self, "output_dir"):
            with open(f"{self.output_dir}/vocab_map.json", "w") as f:
                json.dump(self.vocab_map, f, indent=2, ensure_ascii=False) 
            stats = self.get_vocab_map_stats()
            with open(f"{self.output_dir}/stats.json", "w") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

    def find_posterior(self, text1, text2):
        '''Find the posterior MLE estimate of self.noise_params given text1 and text2
        Args:
            text1: str, text 1
            text2: str, text 2
        Returns:
            MLE estimate of self.theta_global
        '''
        vocab1 = self.get_vocab(text1)
        vocab2 = self.get_vocab(text2)

        # Find the number of words switched out
        switched_out = len(vocab1 - vocab2)
        mle_est_theta_global = switched_out / len(vocab1)

        # OR we can do the following:
        ## For each word, see whether its frequencies are significantly different in text1 and text2
        switched_out = 0
        total1 = sum(vocab1.values())
        total2 = sum(vocab2.values())
        for word in vocab1:

            if vocab1[word] >= 10 and vocab2[word] <= 2: # clear case
                switched_out += 1
                continue
            
            # In the case that word in rare in text1, we can't really say that it was switched out if it is also rare in text2
            # We use the chi-squared test to determine whether the word is significantly rarer in text2 than in text1
            expected = vocab1[word] 
            observed = (vocab2[word] / total2 )* total1 
            # We need to scale the observed frequency to the size of text1 because 
            # the totals over categories need to be the same for the chi-squared test

            # Chi-squared test
            res = chisquare(f_exp=[expected, total1 - expected], f_obs=[observed, total1 - observed])
            if res.pvalue < 0.05 and vocab1[word] > vocab2[word]: # significantly different, and word is rarer in text2
                switched_out += 1

        mle_est_theta_global = switched_out / len(vocab1)

        # OR we can use the absolute frequencies
        switched_out = total = 0
        for word in vocab1:
            if vocab1[word] <= 3:
                continue
            total += 1
            if vocab2[word] <= 3:
                switched_out += 1
        mle_est_theta_global = switched_out / total
        
        return mle_est_theta_global




# class GlobalLexicalNoiser(Noise):
#     '''
#     Noise type: switch out words from the vocabulary with non-words, and apply this change globally to every occurrence.
#     Required params: text_file, theta_global
#     Input format: {param: value}
#     '''
    
#     def __init__(self, noise_params):
#         '''Initialize noise with noise parameters
#         Args:
#             noise_params: dict, noise parameters, like {theta_1: 0.5}
#             Should contain:
#                 text_file: str, txt file
#                 theta_global: float, probability of switching out a word with a non-word
#         '''
#         self.class_name = "GlobalLexicalNoiser"
#         self.required_keys = {"text_file", "theta_global"}
#         # self.required_keys = {"lang", "insert_theta", "delete_theta", "swap_theta"}
#         self.check_noise_params(noise_params)

#         for key in noise_params:
#             setattr(self, key, noise_params[key])

#         if not hasattr(self, "chargram_length"):
#             self.chargram_length = 3

#         # Initialize vocabulary
#         self.vocab = self.get_vocab(self.text_file)
#         self.chargram_models = self.train_chargram_model(self.chargram_length)
#         self.vocab_map = self.construct_new_vocab()
#         self.tag2wordlist = self.get_tag2wordlist()

#     @staticmethod
#     def get_vocab(text_file):
#         '''Initialize vocabulary from vocab file'''
#         print(f"Initializing vocabulary from {text_file}...")
#         vocab = defaultdict(lambda: 0)
#         for line in open(text_file):
#             words = line.strip().split()
#             for word in words:
#                 # Remove punctuation
#                 word = word.strip(".,!?")
#                 # If word has non-alphabetic characters, skip
#                 if not word.isalpha():
#                     continue
#                 vocab[word.lower()] += 1
#         print(f"Finished initializing vocabulary from {text_file}!")

#         return vocab

#     def train_chargram_model(self, chargram_length=3):
#         '''Train a character n-gram model on text
#         Args:
#             n: int, char n-gram length
#         Returns:
#             chargram_models: dict, contains character n-gram models for all n <= chargram_length {n: model} 
#                 - model: defaultdict, character n-gram model of type {prefix: {suffix: count}}
#         '''
#         print(f"Training chargram model with chargram length {chargram_length}...")
#         chargram_models = dict() # contains chargram_models for all cgram lengths <= chargram_length
#         for n in range(1, chargram_length + 1):
#             chargram_model = defaultdict(lambda: defaultdict(lambda: 0)) # {prefix: {suffix: count}}

#             for word in self.vocab:
#                 word = "!" + word # Add start token
#                 for i in range(len(word) - n + 1):
#                     ngram = word[i:i+n]
#                     # Increment count for ngram
#                     chargram_model[ngram[:-1]][ngram[-1]] += 1
        
#             chargram_models[n] = chargram_model

#         print(f"Finished training chargram model with chargram length {chargram_length}!")
#         return chargram_models

#     def generate_word(self, mean_length):
#         '''
#         This function is for generating a non-word using the character n-gram model. We will:
#         1. Sample the length of the non-word from a Poisson centered around mean_length
#         2. Use self.chargram_models to generate the rest of the non-word based on the length of prefix
#         Args:
#             mean_length: float, mean length of non-word
#         '''

#         # Sample length of non-word from Poisson, must be at least 1

#         length = max(1, np.random.poisson(mean_length))
#         length += 1
#         word = "!"

#         while word == "!" or word[1:].lower() in self.vocab:
#             # If generated word in vocab, generate another word
#             word = "!"
#             for _ in range(length):
#                 if len(word) < self.chargram_length-1:
#                     # If the word is shorter than the length of the chargram model, 
#                     # we will apply chargram model of the length of the word
#                     chargram_model = self.chargram_models[len(word) + 1]
#                     prefix = word
#                 else:
#                     # If the word is longer than the prefix of the chargram model, 
#                     # we will apply the chargram model of the length of the model
#                     chargram_model = self.chargram_models[self.chargram_length]
#                     prefix = word[-(self.chargram_length-1):]
#                 while len(list(chargram_model[prefix].keys())) == 0:
#                     # Backing off to shorter chargram models
#                     chargram_model = self.chargram_models[len(prefix)]
#                     prefix = prefix[1:]
#                 # print(list(chargram_model[prefix].keys()))
#                 # print(len(list(chargram_model[prefix].keys())))

#                 # Sample the next character based on the prefix
#                 try:
#                     next_char = np.random.choice(list(chargram_model[prefix].keys()), p=np.array(list(chargram_model[prefix].values())) / sum(chargram_model[prefix].values()))
#                 except:
#                     print(f"Error: {prefix} not in chargram model")
#                     # print(f"Chargram model: {chargram_model}")
#                     print(f"Word: {word}")
#                     print(f"Length: {length}")
#                     print(f"Mean Length: {mean_length}")
#                     print(f"Chargram Length: {self.chargram_length}")
#                     print(f"Prefix: {prefix}")
#                     print(f"{not chargram_model[prefix]}")
#                     print(f"Chargram model: {chargram_model[prefix].keys()}")

#                     raise
#                 word += next_char

#         return word[1:]
    
#     def get_tag2wordlist(self):
#         '''Get tag2wordlist from the JSON file'''
#         with open("ud_closed_class_wordlists/en_ewt-ud-train.json") as f:
#             tag2wordlist = json.load(f)
#         return tag2wordlist

#     def construct_new_vocab(self):
#         '''
#         With probability theta_global, switch out a word from the vocabulary with a non-word
#         Returns:
#             vocab_map: dict, mapping of old word to new word
#         '''
#         vocab_map = dict()
#         for word in self.vocab:
#             if random.random() < self.theta_global:
#                 # print(f"Switching out {word}")
#                 new_word = self.generate_word(len(word))
#                 vocab_map[word] = new_word
#             else:
#                 vocab_map[word] = word
#         return vocab_map


#     def apply_noise(self, input):
#         '''Apply noise to input
#         Args:
#             input: str, input text
#         Returns:
#             str, noised text
#         '''
#         # Apply noise
#         # For each word, map it to the corresponding word using the vocab map self.vocab_map

#         noised_input = list()
#         for input_word in input.split():
#             word = input_word.strip(".,!?").lower()
#             if word in self.vocab_map:
#                 mapped_word = self.vocab_map[word]
#                 # Capitalize first letter if original word was capitalized
#                 if input_word[0].isupper():
#                     mapped_word = mapped_word.capitalize()
#                 # Add punctuation back
#                 if input_word[-1] in ".,!?":
#                     mapped_word += input_word[-1]
#                 noised_input.append(mapped_word)
#             else:
#                 noised_input.append(input_word)
#         return " ".join(noised_input)

#     def record_noiser_artifacts(self):
#         '''Record vocab map, number of words switched out'''
#         with open("vocab.txt", "w") as f:
#             for word in self.vocab:
#                 f.write(word + "\n")
#         raise NotImplementedError


#     def find_posterior(self, text1, text2):
#         '''Find the posterior MLE estimate of self.noise_params given text1 and text2
#         Args:
#             text1: str, text 1
#             text2: str, text 2
#         Returns:
#             MLE estimate of self.theta_global
#         '''
#         vocab1 = self.get_vocab(text1)
#         vocab2 = self.get_vocab(text2)

#         # Find the number of words switched out
#         switched_out = len(vocab1 - vocab2)
#         mle_est_theta_global = switched_out / len(vocab1)

#         # OR we can do the following:
#         ## For each word, see whether its frequencies are significantly different in text1 and text2
#         switched_out = 0
#         total1 = sum(vocab1.values())
#         total2 = sum(vocab2.values())
#         for word in vocab1:

#             if vocab1[word] >= 10 and vocab2[word] <= 2: # clear case
#                 switched_out += 1
#                 continue
            
#             # In the case that word in rare in text1, we can't really say that it was switched out if it is also rare in text2
#             # We use the chi-squared test to determine whether the word is significantly rarer in text2 than in text1
#             expected = vocab1[word] 
#             observed = (vocab2[word] / total2 )* total1 
#             # We need to scale the observed frequency to the size of text1 because 
#             # the totals over categories need to be the same for the chi-squared test

#             # Chi-squared test
#             res = chisquare(f_exp=[expected, total1 - expected], f_obs=[observed, total1 - observed])
#             if res.pvalue < 0.05 and vocab1[word] > vocab2[word]: # significantly different, and word is rarer in text2
#                 switched_out += 1

#         mle_est_theta_global = switched_out / len(vocab1)

#         # OR we can use the absolute frequencies
#         switched_out = total = 0
#         for word in vocab1:
#             if vocab1[word] <= 3:
#                 continue
#             total += 1
#             if vocab2[word] <= 3:
#                 switched_out += 1
#         mle_est_theta_global = switched_out / total
        
#         return mle_est_theta_global

