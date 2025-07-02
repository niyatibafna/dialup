from .noise import Noise
from .phonological import PhonologicalNoiserAugmenter, GlobalPhonologicalNoiser
import random
from collections import defaultdict, Counter
import numpy as np
import json
import sys
import os
from tqdm import tqdm

from .utils.misc import normalize_lang_codes, get_character_set

from .utils.get_functional_words import get_tag2wordlist

from importlib.resources import files as pkg_files

from scipy.stats import chisquare

random.seed(42)
np.random.seed(42)
rng = np.random.default_rng(42)


class MorphologicalNoiserAugmenter(Noise):

    '''
    Noise type: Swap out suffixes of content words with some probability. A changed suffix is swapped globally.
    Required params: text_file, theta_morph_global
    Input format: {param: value}
    '''
    
    def __init__(self, noise_params):
        '''Initialize noise with noise parameters
        Args:
            noise_params: dict, noise parameters, like {theta_1: 0.5}
            Should contain:
                lang: str, language code
                text_file: str, txt file. We'll learn a character ngram model from this text.
                theta_morph_global: float, probability of switching out a suffix
            Also accepts:
                chargram_length: int, character n-gram length (default: 3)
                output_dir: str, output directory for noiser artifacts
        '''

        self.class_name = "MorphologicalNoiserAugmenter"
        self.required_keys = {"lang", "text_file", "theta_morph_global"}
        self.allowed_keys = {"output_dir"}
        self.check_noise_params(noise_params)

        for key in noise_params:
            if key == "lang":
                self.lang = normalize_lang_codes(noise_params[key])
                continue
            setattr(self, key, noise_params[key])
        
        _, self.character_set = get_character_set(self.lang)

        # Initialize vocabulary
        self.vocab = self.get_vocab(self.text_file)
        self.phon_noiser = PhonologicalNoiserAugmenter({"lang": self.lang, "theta_phon": 0.5, "text_file": self.text_file})
        self.tag2wordlist = get_tag2wordlist(self.lang)
        self.suffix_freq, self.most_frequent_word_per_suffix = self.get_suffix_frequency()
        # self.filter_suffix_frequency()
        self.filter_suffix_topk()


        if hasattr(self, "output_dir"):
            os.makedirs(self.output_dir, exist_ok=True)
        

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

            for word, freq in self.vocab.items():
                word = "!" + word # Add start token
                for i in range(len(word) - n + 1):
                    ngram = word[i:i+n]
                    # Increment count for ngram
                    chargram_model[ngram[:-1]][ngram[-1]] += freq
        
            chargram_models[n] = chargram_model

        print(f"Finished training chargram model with chargram length {chargram_length}!")
        return chargram_models
    
    def train_suffix_chargram_model(self, chargram_length=3):
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

            for word, freq in self.suffix_freq.items():
                word = "!" + word # Add start token
                for i in range(len(word) - n + 1):
                    ngram = word[i:i+n]
                    # Increment count for ngram
                    chargram_model[ngram[:-1]][ngram[-1]] += freq
        
            chargram_models[n] = chargram_model

        print(f"Finished training chargram model with chargram length {chargram_length}!")
        return chargram_models

    def generate_word(self, mean_length, init_prefix = ""):
        '''
        This function is for generating a non-word using the character n-gram model. We will:
        1. Sample the length of the non-word from a Poisson centered around mean_length
        2. Use self.chargram_models to generate the rest of the non-word based on the length of prefix

        Note that we are generating suffixes (but it's the same thing in principle).
        Args:
            mean_length: float, mean length of non-word
            prefix: str, prefix of the non-word
        '''

        # Sample length of non-word from Poisson, must be at least 1

        length = max(1, rng.poisson(mean_length))
        length += 1
        word = "!" + init_prefix

        # while word == "!" + init_prefix:
            # If generated word in vocab, generate another word
        # word = "!"
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
                # next_char = rng.choice(list(chargram_model[prefix].keys()), \
                #         p=np.array(list(chargram_model[prefix].values())) / sum(chargram_model[prefix].values()))
                # Pick max char

                next_char = max(chargram_model[prefix], key=chargram_model[prefix].get)

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

        return word[len(init_prefix) + 1:]
    

    def get_vocab(self, text_file):
        '''Initialize vocabulary from vocab file'''
        print(f"Initializing vocabulary from {text_file}...")
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
                # All characters in word must be in character set
                # if not all(char in self.character_set for char in word):
                #     print(f"Not in character set: {word}")
                #     continue
                                
                vocab[word.lower()] += 1
        print(f"Finished initializing vocabulary from {text_file}!")
        print(f"Length of vocab: {len(vocab)}")

        return vocab

    

    def is_word_functional(self, word):
        '''Check if word is functional'''
        for tag in self.tag2wordlist:
            if tag == "AUX".casefold(): # AUX words *can* be affected by morphological change
                continue
            if word in self.tag2wordlist[tag]:
                return True
        return False

    def get_suffix_frequency(self):
        '''Get suffix frequency map from vocab
        Args:
            vocab: dict, vocabulary of type {word: count}
        Returns:
            suffix_freq: dict, contains the frequency of each suffix
            most_frequent_word_per_suffix: dict, contains the most frequent word for each suffix. This is 
                used to condition the new suffix on the stem of the word if the suffix is swapped
        '''
        suffix_freq = defaultdict(lambda: 0)
        most_frequent_word_per_suffix = defaultdict(lambda: ("", 0))
        for word in self.vocab:
            for i in range(1, round(len(word)/2) + 1): #only allow half the word to be a suffix
                suffix_freq[word[-i:]] += self.vocab[word]
                if self.vocab[word] > most_frequent_word_per_suffix[word[-i:]][1]:
                    most_frequent_word_per_suffix[word[-i:]] = (word, self.vocab[word])

        return suffix_freq, most_frequent_word_per_suffix

    def filter_suffix_frequency(self):
        '''
        Filter
        Remove all suffixes with a frequency <= 1
        '''
        self.suffix_freq = {suffix: freq for suffix, freq in self.suffix_freq.items() if freq > 20}
        self.suffix_freq = {suffix: freq for suffix, freq in self.suffix_freq.items() if len(suffix) > 1}
        self.suffix_freq = defaultdict(lambda: 0, self.suffix_freq)

    def filter_suffix_topk(self, k=200):
        '''
        Filter
        Take only top k suffixes
        '''
        lang2k = {"hin": 100, "deu": 120, "fra": 200, "spa": 200, "tur": 400}
        if self.lang in lang2k:
            k = lang2k[self.lang]
        else:
            k = 200

        self.suffix_freq = {suffix: freq for suffix, freq in self.suffix_freq.items() if len(suffix) > 1}
        sorted_suffixes = sorted(self.suffix_freq, key=lambda x: self.suffix_freq[x], reverse=True)
        self.suffix_freq = {suffix: self.suffix_freq[suffix] for suffix in sorted_suffixes[:k]}
        self.suffix_freq = defaultdict(lambda: 0, self.suffix_freq)


    def construct_suffix_map_with_char_lm(self):
        '''
        We'll first sample the suffixes to be swapped based on the log frequency of the suffixes.
        Then we'll map each chosen suffix to a new suffix
        Returns:
            suffix_map: dict, mapping of old suffix to new suffix        
        '''
        suffix_map = {
            suffix: suffix for suffix in self.suffix_freq
        }

        for suffix in suffix_map:
            if random.random() > self.theta_morph_global:
                continue
            most_freq_word = self.most_frequent_word_per_suffix[suffix][0]
            prefix = most_freq_word[:-len(suffix)]
            new_suffix = self.generate_word(len(suffix), init_prefix = prefix)
            suffix_map[suffix] = new_suffix

            # print(f"Suffix: {suffix}")
            # print(f"Most frequent word: {most_freq_word}")
            # print(f"Prefix: {prefix}")
            # print(f"New Suffix: {new_suffix}")

        return suffix_map

    def construct_suffix_map(self):
        '''
        We'll toss a coin for each suffix based on theta_morph_global.
        Then we'll map each chosen suffix to a new suffix
        Returns:
            suffix_map: dict, mapping of old suffix to new suffix        
        '''
        suffix_map = {
            suffix: suffix for suffix in self.suffix_freq
        }

        for suffix in suffix_map:
            if random.random() > self.theta_morph_global:
                continue
            new_suffix = self.phon_noiser.apply_noise_for_sure(suffix)
            suffix_map[suffix] = new_suffix

            # print(f"Suffix: {suffix}")
            # print(f"Most frequent word: {most_freq_word}")
            # print(f"Prefix: {prefix}")
            # print(f"New Suffix: {new_suffix}")

        return suffix_map


    def construct_suffix_map_biased(self):
        '''
        We'll first sample the suffixes to be swapped based on the log frequency of the suffixes.
        Then we'll map each chosen suffix to a new suffix
        Returns:
            suffix_map: dict, mapping of old suffix to new suffix        
        '''
        suffix_map = {
            suffix: suffix for suffix in self.suffix_freq
        }

        # We want suffixes with a higher log frequency to have a higher chance of being swapped, since
        # those are more likely to be actual suffixes
        # We therefore first get log frequencies of suffixes, and use those as weights.
        # Note that if each suffix has a swap probability of theta_morph_global, then the number of swaps
        # is binomial(#suffixes, theta_morph_global)
        # Since we want to sample the suffixes based on their log frequencies, #suffixes = sum(log_freqs) (rounded)
        # We get #swaps = binomial(sum(log_freqs), theta_morph_global)
        # Then we simply sample #swaps suffixes weighted by their log frequencies

        ## Get weights based on log frequency
        suffixes, counts = zip(*self.suffix_freq.items())
        weights = np.log(np.array(counts) + 1)
        norm_weights = weights / sum(weights)
        ## Length of "vocab" i.e. sum of log frequencies
        total_log_freq_vocab_length = round(sum(weights))
        print(f"Total log frequency of vocab length: {total_log_freq_vocab_length}")
        ## Now we'll sample the number of suffixes to be swapped from binomial(total_log_freq_vocab_length, theta_morph_global)
        # num_suffixes = np.random.binomial(total_log_freq_vocab_length, self.theta_morph_global)
        num_suffixes = rng.binomial(len(suffixes), self.theta_morph_global)
        print(f"Number of suffixes to be swapped: {num_suffixes}")
        # Sample the suffixes
        ## We do this with replacement, since sampling without replacement makes no sense to me.
        ## Also because we want theta to basically be some idea of the fraction of suffix mass as such
        ## that gets swapped out. We want more common suffixes to 'take up' more mass if they are sampled
        ## This will happen automatically in sampling with replacement since the more common suffixes will
        ## be sampled more often.
        sampled_suffixes = rng.choice(suffixes, num_suffixes, p=norm_weights, replace=True)

        # Now we'll map each suffix to a new suffix
        ## We would like to condition the new suffix on the stems of the words preceding the suffix.
        ## In practice, we only take the most common such stem and use that as a prefix.
        for suffix in tqdm(set(sampled_suffixes)):
            most_freq_word = self.most_frequent_word_per_suffix[suffix][0]
            prefix = most_freq_word[:-len(suffix)]
            new_suffix = self.generate_word(len(suffix), init_prefix = prefix)
            suffix_map[suffix] = new_suffix

            # print(f"Suffix: {suffix}")
            # print(f"Most frequent word: {most_freq_word}")
            # print(f"Prefix: {prefix}")
            # print(f"New Suffix: {new_suffix}")

        return suffix_map
    

    def construct_new_vocab(self, suffix_map, input):
        '''
        For every word, we'll consider all its suffixes (if present in self.suffix_frequency).
        We'll sample a suffix from the candidates based on log frequency.
        We'll then swap out the suffix with the new suffix.
        '''
        vocab_map = dict()
        for word in input.split():

            # We will not noise functional words: EXCEPT auxiliaries
            # This is the compromise we make, because many languages have morphologically complex 
            # auxiliaries.
            if self.is_word_functional(word):
                vocab_map[word] = word
                continue

            # Get all suffixes of the word
            suffixes = [word[-i:] for i in range(1, len(word))]
            # Get the suffixes that are present in suffix_freq
            suffixes = [suffix for suffix in suffixes if suffix in self.suffix_freq]
            if len(suffixes) == 0:
                vocab_map[word] = word
                continue
            
            ### We could sample a suffix based on frequency BUT this would mean we always
            ### pick the shortest.
            # # Get weights based on log frequency
            # # weights = np.log(np.array([self.suffix_freq[suffix] for suffix in suffixes]) + 1)
            # # Get weights based on log frequency
            # weights = np.array([self.suffix_freq[suffix] for suffix in suffixes])
            # weights = weights / sum(weights)
            # # Sample a suffix
            # sampled_suffix = rng.choice(suffixes, 1, p=weights)[0]
            # Swap out the suffix

            # Instead, let's just pick the longest suffix:
            sampled_suffix = max(suffixes, key=lambda x: len(x))
            new_word = word[:-len(sampled_suffix)] + suffix_map[sampled_suffix]
            vocab_map[word] = new_word
        
        return vocab_map

    def get_suffix_map_stats(self):
        '''
        Get stats of suffix map
        '''
        stats = dict()
        stats["theta_morph_global"] = self.theta_morph_global
        stats["num_suffixes"] = len(self.suffix_freq)
        stats["total_occurrences_of_all_suffixes"] = sum(self.suffix_freq.values())
        stats["total_log_occurrences_of_all_suffixes"] = sum(np.log(np.array(list(self.suffix_freq.values())) + 1))
        return stats
    
   
    def apply_noise(self, input):
        '''Apply noise to input
        Args:
            input: str, input text
        Returns:
            str, noised text
        '''
        # Apply noise

        # Construct suffix --> new suffix map for all suffixes
        suffix_map = self.construct_suffix_map()

        # Construct vocab map for words in input, which maps each word to a new word by swapping out suffixes of content words
        vocab_map = self.construct_new_vocab(suffix_map, input)

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



class GlobalMorphologicalNoiser(Noise):

    '''
    Noise type: Swap out suffixes of content words with some probability. A changed suffix is swapped globally.
    Required params: text_file, theta_morph_global
    Input format: {param: value}
    '''
    
    def __init__(self, noise_params):
        '''Initialize noise with noise parameters
        Args:
            noise_params: dict, noise parameters, like {theta_1: 0.5}
            Should contain:
                text_file: str, txt file. We'll learn a character ngram model from this text.
                theta_morph_global: float, probability of switching out a suffix
            Also accepts:
                chargram_length: int, character n-gram length (default: 3)
                output_dir: str, output directory for noiser artifacts
        '''

        self.class_name = "GlobalMorphologicalNoiser"
        self.required_keys = {"lang", "text_file", "theta_morph_global"}
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

        # Initialize vocabulary
        self.vocab = self.get_vocab(self.text_file)
        ### We're not using chargram models for now
        # self.chargram_models = self.train_chargram_model(self.chargram_length)
        self.phon_noiser = GlobalPhonologicalNoiser({"lang": self.lang, "theta_phon": 0.5, "text_file": self.text_file})
        self.tag2wordlist = get_tag2wordlist(self.lang)
        self.suffix_freq, self.most_frequent_word_per_suffix = self.get_suffix_frequency()
        # self.filter_suffix_frequency()
        self.filter_suffix_topk()
        # self.chargram_models = self.train_suffix_chargram_model(self.chargram_length)

        # Construct suffix --> new suffix map
        self.suffix_map = self.construct_suffix_map()

        # Construct vocab map, which maps each word to a new word by swapping out suffixes of content words
        self.vocab_map = self.construct_new_vocab()

        if hasattr(self, "output_dir"):
            os.makedirs(self.output_dir, exist_ok=True)
        

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

            for word, freq in self.vocab.items():
                word = "!" + word # Add start token
                for i in range(len(word) - n + 1):
                    ngram = word[i:i+n]
                    # Increment count for ngram
                    chargram_model[ngram[:-1]][ngram[-1]] += freq
        
            chargram_models[n] = chargram_model

        print(f"Finished training chargram model with chargram length {chargram_length}!")
        return chargram_models
    
    def train_suffix_chargram_model(self, chargram_length=3):
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

            for word, freq in self.suffix_freq.items():
                word = "!" + word # Add start token
                for i in range(len(word) - n + 1):
                    ngram = word[i:i+n]
                    # Increment count for ngram
                    chargram_model[ngram[:-1]][ngram[-1]] += freq
        
            chargram_models[n] = chargram_model

        print(f"Finished training chargram model with chargram length {chargram_length}!")
        return chargram_models

    def generate_word(self, mean_length, init_prefix = ""):
        '''
        This function is for generating a non-word using the character n-gram model. We will:
        1. Sample the length of the non-word from a Poisson centered around mean_length
        2. Use self.chargram_models to generate the rest of the non-word based on the length of prefix

        Note that we are generating suffixes (but it's the same thing in principle).
        Args:
            mean_length: float, mean length of non-word
            prefix: str, prefix of the non-word
        '''

        # Sample length of non-word from Poisson, must be at least 1

        length = max(1, rng.poisson(mean_length))
        length += 1
        word = "!" + init_prefix

        # while word == "!" + init_prefix:
            # If generated word in vocab, generate another word
        # word = "!"
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
                # next_char = rng.choice(list(chargram_model[prefix].keys()), \
                #         p=np.array(list(chargram_model[prefix].values())) / sum(chargram_model[prefix].values()))
                # Pick max char

                next_char = max(chargram_model[prefix], key=chargram_model[prefix].get)

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

        return word[len(init_prefix) + 1:]
    

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
                # if not all(char in self.character_set for char in word):
                #     print(f"Not in character set: {word}")
                #     continue
                                
                vocab[word.lower()] += 1
        print(f"Finished initializing vocabulary from {text_file}!")
        print(f"Length of vocab: {len(vocab)}")

        return vocab


    def is_word_functional(self, word):
        '''Check if word is functional'''
        for tag in self.tag2wordlist:
            if tag == "AUX".casefold(): # AUX words *can* be affected by morphological change
                continue
            if word in self.tag2wordlist[tag]:
                return True
        return False

    def get_suffix_frequency(self):
        '''Get suffix frequency map from vocab
        Args:
            vocab: dict, vocabulary of type {word: count}
        Returns:
            suffix_freq: dict, contains the frequency of each suffix
            most_frequent_word_per_suffix: dict, contains the most frequent word for each suffix. This is 
                used to condition the new suffix on the stem of the word if the suffix is swapped
        '''
        suffix_freq = defaultdict(lambda: 0)
        most_frequent_word_per_suffix = defaultdict(lambda: ("", 0))
        for word in self.vocab:
            for i in range(1, round(len(word)/2) + 1): #only allow half the word to be a suffix
                suffix_freq[word[-i:]] += self.vocab[word]
                if self.vocab[word] > most_frequent_word_per_suffix[word[-i:]][1]:
                    most_frequent_word_per_suffix[word[-i:]] = (word, self.vocab[word])

        return suffix_freq, most_frequent_word_per_suffix

    def filter_suffix_frequency(self):
        '''
        Filter
        Remove all suffixes with a frequency <= 1
        '''
        self.suffix_freq = {suffix: freq for suffix, freq in self.suffix_freq.items() if freq > 20}
        self.suffix_freq = {suffix: freq for suffix, freq in self.suffix_freq.items() if len(suffix) > 1}
        self.suffix_freq = defaultdict(lambda: 0, self.suffix_freq)

    def filter_suffix_topk(self, k=200):
        '''
        Filter
        Take only top k suffixes
        '''
        lang2k = {"hin": 100, "deu": 120, "fra": 200, "spa": 200}
        if self.lang in lang2k:
            k = lang2k[self.lang]
        else:
            k = 200

        self.suffix_freq = {suffix: freq for suffix, freq in self.suffix_freq.items() if len(suffix) > 1}
        sorted_suffixes = sorted(self.suffix_freq, key=lambda x: self.suffix_freq[x], reverse=True)
        self.suffix_freq = {suffix: self.suffix_freq[suffix] for suffix in sorted_suffixes[:k]}
        self.suffix_freq = defaultdict(lambda: 0, self.suffix_freq)


    def construct_suffix_map_with_char_lm(self):
        '''
        We'll first sample the suffixes to be swapped based on the log frequency of the suffixes.
        Then we'll map each chosen suffix to a new suffix
        Returns:
            suffix_map: dict, mapping of old suffix to new suffix        
        '''
        suffix_map = {
            suffix: suffix for suffix in self.suffix_freq
        }

        for suffix in tqdm(suffix_map):
            if random.random() > self.theta_morph_global:
                continue
            most_freq_word = self.most_frequent_word_per_suffix[suffix][0]
            prefix = most_freq_word[:-len(suffix)]
            new_suffix = self.generate_word(len(suffix), init_prefix = prefix)
            suffix_map[suffix] = new_suffix

            # print(f"Suffix: {suffix}")
            # print(f"Most frequent word: {most_freq_word}")
            # print(f"Prefix: {prefix}")
            # print(f"New Suffix: {new_suffix}")

        return suffix_map

    def construct_suffix_map(self):
        '''
        We'll toss a coin for each suffix based on theta_morph_global.
        Then we'll map each chosen suffix to a new suffix
        Returns:
            suffix_map: dict, mapping of old suffix to new suffix        
        '''
        suffix_map = {
            suffix: suffix for suffix in self.suffix_freq
        }

        for suffix in tqdm(suffix_map):
            if random.random() > self.theta_morph_global:
                continue
            new_suffix = self.phon_noiser.apply_noise_for_sure(suffix)
            suffix_map[suffix] = new_suffix

            # print(f"Suffix: {suffix}")
            # print(f"Most frequent word: {most_freq_word}")
            # print(f"Prefix: {prefix}")
            # print(f"New Suffix: {new_suffix}")

        return suffix_map


    def construct_suffix_map_biased(self):
        '''
        We'll first sample the suffixes to be swapped based on the log frequency of the suffixes.
        Then we'll map each chosen suffix to a new suffix
        Returns:
            suffix_map: dict, mapping of old suffix to new suffix        
        '''
        suffix_map = {
            suffix: suffix for suffix in self.suffix_freq
        }

        # We want suffixes with a higher log frequency to have a higher chance of being swapped, since
        # those are more likely to be actual suffixes
        # We therefore first get log frequencies of suffixes, and use those as weights.
        # Note that if each suffix has a swap probability of theta_morph_global, then the number of swaps
        # is binomial(#suffixes, theta_morph_global)
        # Since we want to sample the suffixes based on their log frequencies, #suffixes = sum(log_freqs) (rounded)
        # We get #swaps = binomial(sum(log_freqs), theta_morph_global)
        # Then we simply sample #swaps suffixes weighted by their log frequencies

        ## Get weights based on log frequency
        suffixes, counts = zip(*self.suffix_freq.items())
        weights = np.log(np.array(counts) + 1)
        norm_weights = weights / sum(weights)
        ## Length of "vocab" i.e. sum of log frequencies
        total_log_freq_vocab_length = round(sum(weights))
        print(f"Total log frequency of vocab length: {total_log_freq_vocab_length}")
        ## Now we'll sample the number of suffixes to be swapped from binomial(total_log_freq_vocab_length, theta_morph_global)
        # num_suffixes = np.random.binomial(total_log_freq_vocab_length, self.theta_morph_global)
        num_suffixes = rng.binomial(len(suffixes), self.theta_morph_global)
        print(f"Number of suffixes to be swapped: {num_suffixes}")
        # Sample the suffixes
        ## We do this with replacement, since sampling without replacement makes no sense to me.
        ## Also because we want theta to basically be some idea of the fraction of suffix mass as such
        ## that gets swapped out. We want more common suffixes to 'take up' more mass if they are sampled
        ## This will happen automatically in sampling with replacement since the more common suffixes will
        ## be sampled more often.
        sampled_suffixes = rng.choice(suffixes, num_suffixes, p=norm_weights, replace=True)

        # Now we'll map each suffix to a new suffix
        ## We would like to condition the new suffix on the stems of the words preceding the suffix.
        ## In practice, we only take the most common such stem and use that as a prefix.
        for suffix in tqdm(set(sampled_suffixes)):
            most_freq_word = self.most_frequent_word_per_suffix[suffix][0]
            prefix = most_freq_word[:-len(suffix)]
            new_suffix = self.generate_word(len(suffix), init_prefix = prefix)
            suffix_map[suffix] = new_suffix

            # print(f"Suffix: {suffix}")
            # print(f"Most frequent word: {most_freq_word}")
            # print(f"Prefix: {prefix}")
            # print(f"New Suffix: {new_suffix}")

        return suffix_map
    

    def construct_new_vocab(self):
        '''
        For every word, we'll consider all its suffixes (if present in self.suffix_frequency).
        We'll sample a suffix from the candidates based on log frequency.
        We'll then swap out the suffix with the new suffix.
        '''
        vocab_map = dict()
        for word in self.vocab:

            # We will not noise functional words: EXCEPT auxiliaries
            # This is the compromise we make, because many languages have morphologically complex 
            # auxiliaries.
            if self.is_word_functional(word):
                vocab_map[word] = word
                continue

            # Get all suffixes of the word
            suffixes = [word[-i:] for i in range(1, len(word))]
            # Get the suffixes that are present in suffix_freq
            suffixes = [suffix for suffix in suffixes if suffix in self.suffix_freq]
            if len(suffixes) == 0:
                vocab_map[word] = word
                continue
            
            ### We could sample a suffix based on frequency BUT this would mean we always
            ### pick the shortest.
            # # Get weights based on log frequency
            # # weights = np.log(np.array([self.suffix_freq[suffix] for suffix in suffixes]) + 1)
            # # Get weights based on log frequency
            # weights = np.array([self.suffix_freq[suffix] for suffix in suffixes])
            # weights = weights / sum(weights)
            # # Sample a suffix
            # sampled_suffix = rng.choice(suffixes, 1, p=weights)[0]
            # Swap out the suffix

            # Instead, let's just pick the longest suffix:
            sampled_suffix = max(suffixes, key=lambda x: len(x))
            new_word = word[:-len(sampled_suffix)] + self.suffix_map[sampled_suffix]
            vocab_map[word] = new_word
        
        return vocab_map

    def get_suffix_map_stats(self):
        '''
        Get stats of suffix map
        '''
        stats = dict()
        stats["theta_morph_global"] = self.theta_morph_global
        stats["num_suffixes"] = len(self.suffix_freq)
        stats["total_occurrences_of_all_suffixes"] = sum(self.suffix_freq.values())
        stats["total_log_occurrences_of_all_suffixes"] = sum(np.log(np.array(list(self.suffix_freq.values())) + 1))
        stats["num_suffixes_swapped"] = len([suffix for suffix in self.suffix_map if self.suffix_map[suffix] != suffix])
        return stats
    
    def get_vocab_map_stats(self):
        '''
        Get stats of vocab map
        '''
        stats = dict()
        stats["theta_morph_global"] = self.theta_morph_global
        stats["vocab_size"] = len(self.vocab)
        stats["content_words"] = len([word for word in self.vocab if not self.is_word_functional(word)])
        stats["func_words"] = len([word for word in self.vocab if self.is_word_functional(word)])
        stats["content_words_noised_frac"] = round(len([word for word in self.vocab_map if not self.is_word_functional(word) \
                                                 and self.vocab_map[word] != word]) / stats["content_words"], 2)
        return stats
   
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
            with open(f"{self.output_dir}/morph_suffix_map.json", "w") as f:
                json.dump(self.suffix_map, f, indent=2, ensure_ascii=False)
            stats = self.get_suffix_map_stats()
            with open(f"{self.output_dir}/morph_suffix_stats.json", "w") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            with open(f"{self.output_dir}/morph_vocab_map.json", "w") as f:
                json.dump(self.vocab_map, f, indent=2, ensure_ascii=False) 
            stats = self.get_vocab_map_stats()
            with open(f"{self.output_dir}/morph_vocab_stats.json", "w") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            # Record suffix freq
            with open(f"{self.output_dir}/suffix_freq.json", "w") as f:
                sorted_suffixes = sorted(self.suffix_freq, key=lambda x: self.suffix_freq[x], reverse=True)
                json.dump({suffix: self.suffix_freq[suffix] for suffix in sorted_suffixes}, f, indent=2, ensure_ascii=False)


    def find_posterior(self, text1, text2):
        '''Find the posterior MLE estimate of self.noise_params given text1 and text2
        Args:
            text1: str, text 1
            text2: str, text 2
        Returns:
            MLE estimate of self.theta_global
        '''
        # raise NotImplementedError("find_posterior is not implemented for GlobalMorphologicalNoiser")
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



