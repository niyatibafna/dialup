from .noise import Noise
import random
import os
from collections import defaultdict
import json
from .utils.misc import normalize_lang_codes, get_character_set, identify_script, ipa_char_maps, get_equivalence_classes_ipa

random.seed(42)

class PhonologicalNoiserAugmenter(Noise):
    def __init__(self, noise_params):
        '''Initialize phonological noiser with noise parameters
        Args:
            noise_params: dict, noise parameters, like {theta_1: 0.5}
            Should contain:
                lang: str, language code
                text_file: str, path to text file
                theta_phon: float, probability of phonological noise
            Also accepts:
                output_dir: str, path to output directory

        '''
        self.class_name = "PhonologicalNoiserAugmenter"
        self.required_keys = {"lang", "text_file", "theta_phon"}
        self.allowed_keys = {"output_dir"}
        self.check_noise_params(noise_params)

        for key in noise_params:
            setattr(self, key, noise_params[key])

        self.lang = normalize_lang_codes(self.lang)
        script, self.character_set = get_character_set(self.lang)

        # Creates a mapping from each character to a set of characters that are equivalent to it in the script
        self.target_chars = self.create_equivalence_set_for_script_chars()
        self.filter_target_chars()

        if hasattr(self, "output_dir"):
            os.makedirs(self.output_dir, exist_ok=True)

    def is_valid_word(self, word):
        '''Check if word is valid
        Args:
            word: str, word to check
        Returns:
            bool, whether word is valid
        '''
        return all(char in self.character_set for char in word)

    def get_ngrams_from_text(self, input):
        '''Get words from text file
        Returns:
            list, list of words
        '''
        words = {"<"+word+">" for word in input.split() if self.is_valid_word(word)} # Add < and > to denote start and end of word
        
        ngrams = defaultdict(lambda: 0)
        n = 3
        for word in words:
            for i in range(len(word) - n + 1):
                ngram = word[i:i+n]
                ngrams[ngram] += 1
        return ngrams
    
    def sample_new_char_at_random(self, char):
        '''Sample a new character from the character set, maintaining casing
        Args:
            char: str, character to sample new character for
        Returns:
            str, new character, maintain casing
        '''

        new_char = random.choice(list(self.character_set - {char.lower()} - {char.upper()}))
        if char.isupper():
            new_char = new_char.upper()
        if char.islower():
            new_char = new_char.lower()
        return new_char
    
    def sample_new_char(self, char):
        '''Samples a new character from the phonologically motivated target set of the input character.
        If the character is in the target set, samples from the target set. 

        Args:
            char: str, character to sample new character for
        Returns:
            str, new character, maintain casing
        '''
        if len(self.target_chars[char]) != 0:

            new_char = random.choice(list(self.target_chars[char.lower()]))
        
            if char.isupper():
                new_char = new_char.upper()
            if char.islower():
                new_char = new_char.lower()
            # print(f"Swapping {char} with {new_char}")
            return new_char
        
        return char


    def create_equivalence_set_for_script_chars(self):
        '''Create equivalence set for script characters.
        We do this by first mapping into IPA, using equivalence classes that we already have for IPA, and then mapping back to script.
        The returned target set for each char does *not* include the char itself.
        Returns:
            dict, {char: set of chars}
        '''
        ipa_to_script_chars, script_to_ipa_chars = ipa_char_maps()
        self.ipa_to_script_chars = ipa_to_script_chars[self.lang]
        self.script_to_ipa_chars = script_to_ipa_chars[self.lang]
        _, _, equivalence_classes_ipa_per_char = get_equivalence_classes_ipa()
        self.equivalence_classes_ipa_per_char = equivalence_classes_ipa_per_char
        
        target_chars = defaultdict(lambda: set())
        for char in self.character_set:

            if char in self.script_to_ipa_chars:
                ipa_set = self.script_to_ipa_chars[char] # set of ipa characters that char maps to                    
                for ipa_char in ipa_set:
                    # Get equivalence classes for ipa_char
                    ipa_eq_chars = self.equivalence_classes_ipa_per_char[ipa_char] # set of ipa characters that are equivalent to ipa_char
                    for ipa_eq_char in ipa_eq_chars:
                        script_eq_chars = self.ipa_to_script_chars[ipa_eq_char] # set of script characters that are equivalent to ipa_eq_char
                        target_chars[char].update(script_eq_chars)
                        # if char == "r":
                        #     print(ipa_set)
                        #     print(f"ipa char: {ipa_char}")
                        #     print(f"ipa_eq_chars: {ipa_eq_chars}")
                        #     print(f"script_eq_chars: {script_eq_chars}")
                        #     print(f"target_chars: {target_chars[char]}")
                        

        for char in self.character_set:
            target_chars[char] = target_chars[char] - {char}

        # print(f"Target Chars: {target_chars}")
        # # Pretty print target_chars
        # for char in target_chars:
        #     target_chars[char] = list(target_chars[char])
        #     target_chars[char].sort()
        #     print(f"CHAR: {char}, TARGET CHARS: {target_chars[char]}")

        #     print("\n\n")

        return target_chars

    def filter_target_chars(self):
        '''
        Here, we remove certain characters from the target set, because they're rare in the script
        '''
        # Devanagari
        exclude_set_dev = {"क़", "ख़", "ग़", "ज़", "ड़", "ढ़", "फ़", "य़", "ॠ", "ॡ", "ॢ", "ॣ", "।", "॥", \
                           "०", "१", "२", "३", "४", "५", "६", "७", "८", "९", "॰", "ॱ", "ॲ", "ॳ", "ॴ", "ॵ", "ॶ", "ॷ", "ॸ", "ॹ", "ॺ", "ॻ", "ॼ", "ॽ", "ॾ", "ॿ",\
                            "ञ", "ङ"}
        exclude_set_latin = {"ÿ"}
        exclude_set_arabic = {"'َ'", "'ِ'", "'ٍ'", "'ٍ'", "'ً'", "'ّ'", 'ة'}

        exclude_set = exclude_set_dev.union(exclude_set_latin).union(exclude_set_arabic)
        for char, target_set in self.target_chars.items():
            self.target_chars[char] = target_set - exclude_set


    def construct_charmap_with_context(self, input):
        '''
        Samples source characters given context to swap out globally, and creates a map.
        '''
        chargram_map = {}
        ngrams = self.get_ngrams_from_text(input)
        for ngram in ngrams:
            if random.random() < self.theta_phon:
                # We'll swap out the middle character
                new_char = self.sample_new_char(ngram[1])
                chargram_map[ngram] = ngram[0] + new_char + ngram[2]
            else:
                chargram_map[ngram] = ngram
        
        # print(f"Chargram Map: {chargram_map}")
        return chargram_map

    def apply_noise(self, input):
        '''Apply phonological noise to input
        Args:
            input: str, input text, or list of input texts
        Returns:
            str, noised text
        '''
        # Apply noise
        if isinstance(input, str):
            words = input.split()
        else:
            words = input
        
        chargram_map = self.construct_charmap_with_context(input)

        noised_words = list()
        for word in words:
            if word[0].isupper():
                # We do not affect proper nouns
                noised_words.append(word)
                continue
            noised_word = ""
            if not self.is_valid_word(word):
                noised_words.append(word)
                continue
            word = "<" + word + ">" # Add < and > to denote start and end of word
            for i in range(len(word)):
                if i == 0 or i == len(word) - 1:
                    pass # Don't add EOS and BOS chars
                else:
                    ngram = word[i-1:i+2]
                    if ngram in chargram_map:
                        noised_word += chargram_map[ngram][1]
                    else:
                        noised_word += word[i]

            noised_words.append(noised_word)
        return " ".join(noised_words)


    def apply_noise_for_sure(self, input):
        '''
        Change at least one character in the input for sure
        First, we apply noise as per usual. If nothing changes,
        we use the IPA maps directly to swap out at least one character
        Note: This function is useful for the purposes of morphological noising, 
        when we want a guaranteed change in the sampled suffixes.
        Args:
            input: str, input text
        Returns:
            str, noised text
        '''
        noised = self.apply_noise(input)
        if noised != input:
            return noised

        # print(f"SWAPPING SOMETHING: {input}")
        noised_word = ""
        if not self.is_valid_word(input):
            return input
        for idx, c in enumerate(input):
            new_char = self.sample_new_char(c)
            if c != new_char:
                noised_word = input[:idx] + new_char + input[idx+1:]

                # print(f"Swapped: {noised_word}")
                return noised_word
        
        # If we reach here, we haven't changed anything
        idx = random.randint(0, len(input) - 1)
        new_char = self.sample_new_char_at_random(input[idx])
        noised_word = input[:idx] + new_char + input[idx+1:]
        # print(f"Swapped: {noised_word}")
        assert noised_word != input
        return noised_word
            
class GlobalPhonologicalNoiser(Noise):

    def __init__(self, noise_params):
        '''Initialize phonological noiser with noise parameters
        Args:
            noise_params: dict, noise parameters, like {theta_1: 0.5}
            Should contain:
                lang: str, language code
                text_file: str, path to text file
                theta_phon: float, probability of phonological noise
            Also accepts:
                output_dir: str, path to output directory

        '''
        self.class_name = "GlobalPhonologicalNoiser"
        self.required_keys = {"lang", "text_file", "theta_phon"}
        self.allowed_keys = {"output_dir"}
        self.check_noise_params(noise_params)

        for key in noise_params:
            setattr(self, key, noise_params[key])

        self.lang = normalize_lang_codes(self.lang)
        script, self.character_set = get_character_set(self.lang)

        # Creates a mapping from each character to a set of characters that are equivalent to it in the script
        self.target_chars = self.create_equivalence_set_for_script_chars()
        self.filter_target_chars()

        self.chargram_map = self.construct_charmap_with_context()

        # For recording purposes
        self.vocab_map = dict()

        if hasattr(self, "output_dir"):
            os.makedirs(self.output_dir, exist_ok=True)

    def is_valid_word(self, word):
        '''Check if word is valid
        Args:
            word: str, word to check
        Returns:
            bool, whether word is valid
        '''
        return all(char in self.character_set for char in word)

    def get_ngrams_from_text(self):
        '''Get words from text file
        Returns:
            list, list of words
        '''
        words = set()
        with open(self.text_file, "r") as f:
            for line in f:
                line_words = line.strip().split()
                words.update({"<"+word+">" for word in line_words if self.is_valid_word(word)}) # Add < and > to denote start and end of word
        
        ngrams = defaultdict(lambda: 0)
        n = 3
        for word in words:
            for i in range(len(word) - n + 1):
                ngram = word[i:i+n]
                ngrams[ngram] += 1
        return ngrams
    
    def sample_new_char_at_random(self, char):
        '''Sample a new character
        Args:
            char: str, character to sample new character for
        Returns:
            str, new character, maintain casing
        '''

        new_char = random.choice(list(self.character_set - {char.lower()} - {char.upper()}))
        if char.isupper():
            new_char = new_char.upper()
        if char.islower():
            new_char = new_char.lower()
        return new_char
    
    def sample_new_char(self, char):
        '''Samples a new character. If the character is in the target set, samples from the target set. 

        Args:
            char: str, character to sample new character for
        Returns:
            str, new character, maintain casing
        '''
        if len(self.target_chars[char]) != 0:

            new_char = random.choice(list(self.target_chars[char.lower()]))
        
            if char.isupper():
                new_char = new_char.upper()
            if char.islower():
                new_char = new_char.lower()
            # print(f"Swapping {char} with {new_char}")
            return new_char
        
        return char


    def create_equivalence_set_for_script_chars(self):
        '''Create equivalence set for script characters.
        We do this by first mapping into IPA, using equivalence classes that we already have for IPA, and then mapping back to script.
        The returned target set for each char does *not* include the char itself.
        Returns:
            dict, {char: set of chars}
        '''
        ipa_to_script_chars, script_to_ipa_chars = ipa_char_maps()
        self.ipa_to_script_chars = ipa_to_script_chars[self.lang]
        self.script_to_ipa_chars = script_to_ipa_chars[self.lang]
        _, _, equivalence_classes_ipa_per_char = get_equivalence_classes_ipa()
        self.equivalence_classes_ipa_per_char = equivalence_classes_ipa_per_char
        
        target_chars = defaultdict(lambda: set())
        for char in self.character_set:

            if char in self.script_to_ipa_chars:
                ipa_set = self.script_to_ipa_chars[char] # set of ipa characters that char maps to                    
                for ipa_char in ipa_set:
                    # Get equivalence classes for ipa_char
                    ipa_eq_chars = self.equivalence_classes_ipa_per_char[ipa_char] # set of ipa characters that are equivalent to ipa_char
                    for ipa_eq_char in ipa_eq_chars:
                        script_eq_chars = self.ipa_to_script_chars[ipa_eq_char] # set of script characters that are equivalent to ipa_eq_char
                        target_chars[char].update(script_eq_chars)
                        # if char == "r":
                        #     print(ipa_set)
                        #     print(f"ipa char: {ipa_char}")
                        #     print(f"ipa_eq_chars: {ipa_eq_chars}")
                        #     print(f"script_eq_chars: {script_eq_chars}")
                        #     print(f"target_chars: {target_chars[char]}")
                        

        for char in self.character_set:
            target_chars[char] = target_chars[char] - {char}

        # print(f"Target Chars: {target_chars}")
        # # Pretty print target_chars
        # for char in target_chars:
        #     target_chars[char] = list(target_chars[char])
        #     target_chars[char].sort()
        #     print(f"CHAR: {char}, TARGET CHARS: {target_chars[char]}")

        #     print("\n\n")

        return target_chars

    def filter_target_chars(self):
        '''
        Here, we remove certain characters from the target set, because they're rare in the script
        '''
        # Devanagari
        exclude_set_dev = {"क़", "ख़", "ग़", "ज़", "ड़", "ढ़", "फ़", "य़", "ॠ", "ॡ", "ॢ", "ॣ", "।", "॥", \
                           "०", "१", "२", "३", "४", "५", "६", "७", "८", "९", "॰", "ॱ", "ॲ", "ॳ", "ॴ", "ॵ", "ॶ", "ॷ", "ॸ", "ॹ", "ॺ", "ॻ", "ॼ", "ॽ", "ॾ", "ॿ",\
                            "ञ", "ङ"}
        exclude_set_latin = {"ÿ"}
        exclude_set_arabic = {"'َ'", "'ِ'", "'ٍ'", "'ٍ'", "'ً'", "'ّ'", 'ة'}

        exclude_set = exclude_set_dev.union(exclude_set_latin).union(exclude_set_arabic)
        for char, target_set in self.target_chars.items():
            self.target_chars[char] = target_set - exclude_set

                
    def construct_charmap_with_context(self):
        '''
        Samples source characters given context to swap out globally, and creates a map.
        '''
        chargram_map = {}
        ngrams = self.get_ngrams_from_text()
        for ngram in ngrams:
            if random.random() < self.theta_phon:
                # We'll swap out the middle character
                new_char = self.sample_new_char(ngram[1])
                if new_char == ngram[1]:
                    new_char = '0'
                chargram_map[ngram] = ngram[0] + new_char + ngram[2]
            else:
                chargram_map[ngram] = ngram
        
        # print(f"Chargram Map: {chargram_map}")
        return chargram_map


    def construct_charmap(self):
        '''
        Samples source characters to swap out globally, and creates a map.
        Initializes self.charmap: dict, {source_char: target_char}
        '''
        charmap = {}
        for char in self.character_set:
            if random.random() < self.theta_phon:
                charmap[char] = random.choice(list(self.character_set - {char}))
            else:
                charmap[char] = char
        return charmap

    
    def apply_noise(self, input):
        '''Apply phonological noise to input
        Args:
            input: str, input text, or list of input texts
        Returns:
            str, noised text
        '''
        # Apply noise
        if isinstance(input, str):
            words = input.split()
        else:
            words = input
        noised_words = list()
        for word in words:
            if word[0].isupper():
                # We do not affect proper nouns
                noised_words.append(word)
                continue
            noised_word = ""
            if not self.is_valid_word(word):
                noised_words.append(word)
                continue
            word = "<" + word + ">" # Add < and > to denote start and end of word
            for i in range(len(word)):
                if i == 0 or i == len(word) - 1:
                    pass # Don't add EOS and BOS chars
                else:
                    ngram = word[i-1:i+2]
                    if ngram in self.chargram_map:
                        if self.chargram_map[ngram][1] != '0':
                            noised_word += self.chargram_map[ngram][1]
                        else:
                            pass # Don't add the middle character - delete
                    else:
                        noised_word += word[i]

            self.vocab_map[word[1:-1]] = noised_word # All input words go through this function
            noised_words.append(noised_word)
        return " ".join(noised_words)

    def apply_noise_for_sure(self, input):
        '''
        Change at least one character in the input for sure
        First, we apply noise as per usual. If nothing changes,
        we use the IPA maps directly to swap out at least one character
        Args:
            input: str, input text
        Returns:
            str, noised text
        '''
        noised = self.apply_noise(input)
        if noised != input:
            return noised

        # print(f"SWAPPING SOMETHING: {input}")
        noised_word = ""
        if not self.is_valid_word(input):
            return input
        for idx, c in enumerate(input):
            new_char = self.sample_new_char(c)
            if c != new_char:
                noised_word = input[:idx] + new_char + input[idx+1:]

                # print(f"Swapped: {noised_word}")
                return noised_word
        
        # If we reach here, we haven't changed anything
        idx = random.randint(0, len(input) - 1)
        new_char = self.sample_new_char_at_random(input[idx])
        noised_word = input[:idx] + new_char + input[idx+1:]
        # print(f"Swapped: {noised_word}")
        assert noised_word != input
        return noised_word
            

    def get_vocab_chargram_map_stats(self):
        '''Recording purposes'''
        stats = dict()
        stats["theta_phon"] = self.theta_phon
        # Number of words changed 
        stats["num_words_changed"] = len([word for word in self.vocab_map if self.vocab_map[word] != word])
        stats["total_words"] = len(self.vocab_map)
        stats["frac_noised_words"] = stats["num_words_changed"] / len(self.vocab_map)
        # Number of chargrams changed
        stats["num_chargrams_changed"] = len([cgram for cgram in self.chargram_map if self.chargram_map[cgram] != cgram])
        stats["total_chargrams"] = len(self.chargram_map)
        stats["frac_noised_chargrams"] = stats["num_chargrams_changed"] / stats["total_chargrams"]
        
        return stats


    
    def record_noiser_artifacts(self):
        '''Record vocab map, number of words switched out'''

        if hasattr(self, "output_dir"):
            with open(f"{self.output_dir}/phon_source_to_target_set.json", "w") as f:
                # Serialize
                source_to_target_set = {source: list(target_set) for source, target_set in self.target_chars.items()}
                json.dump(source_to_target_set, f, indent=2, ensure_ascii=False)

            with open(f"{self.output_dir}/phon_chargram_map.json", "w") as f:
                json.dump(self.chargram_map, f, indent=2, ensure_ascii=False)
            
            with open(f"{self.output_dir}/phon_vocab_map.json", "w") as f:
                json.dump(self.vocab_map, f, indent=2, ensure_ascii=False)

            stats = self.get_vocab_chargram_map_stats()
            with open(f"{self.output_dir}/phon_stats.json", "w") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def find_posterior(self, text1, text2):
        '''Find the posterior MLE estimate of self.noise_params given text1 and text2
        Args:
            text1: str, text 1
            text2: str, text 2
        Returns:
            MLE estimate of self.noise_params (dict)
        '''
        raise NotImplementedError
    