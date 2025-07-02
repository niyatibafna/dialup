import random
from .utils.get_functional_words import get_tag2wordlist
random.seed(42)

class Denoise:
    def __init__(self, lrl, hrl, bilingual_lexicon):
        self.lrl = lrl
        self.hrl = hrl
        self.tag2wordlist = get_tag2wordlist(hrl)
        self.lrl_to_hrl = bilingual_lexicon
        
        
    def is_hrl_word_functional(self, word):
        '''Check if word is functional'''
        for tag in self.tag2wordlist:
            if word in self.tag2wordlist[tag]:
                return True
        return False
    
    def is_lrl_word_functional(self, word, k=1):
        hrl_equivalent = self.get_top_equivalent(word)
        if not hrl_equivalent: # in the case that there is no hrl equivalent
            return None, False
        if self.is_hrl_word_functional(hrl_equivalent):
            return hrl_equivalent, True
        return None, False
    
    
    def is_number(self, s):
        try:
            complex(s) # for int, long, float and complex
        except ValueError:
            return False
        return True
    
    def get_all_equivalents(self, word):
        '''Get all translations of a word'''
        if self.is_number(word):
            return [word]
        if word in self.lrl_to_hrl.keys():
            entries = self.lrl_to_hrl[word]
            # Sort keys by length in descending order
            sorted_keys = sorted(entries.keys(), key=len, reverse=True)
            return sorted_keys
        return None
        
    def get_top_equivalent(self, word):
        '''Get the best translation of a word'''
        if self.is_number(word):
            return word
        if word in self.lrl_to_hrl.keys():
            entries = self.lrl_to_hrl[word]
            if not isinstance(entries, dict):
                return None
            max_score = max(value for value in entries.values() if isinstance(value, (int, float)))
            top_hrl_words = []
            for hrl_word, score in entries.items():
                if score == max_score:
                    top_hrl_words.append(hrl_word)
            return random.choice(top_hrl_words)
        else:
            return None
    
    def preprocess(self, input_word, punct=".,!?।\"”"):
        no_punct = input_word.strip(punct)
        raw = no_punct.lower()
        return no_punct, raw
    
    def maintain_casing(self, no_punct, mapped_word):
        # sometimes the translations add in their own punctuation.
        # The line below removes punctuation.
        mapped_word = mapped_word.strip(".,!?।\"”")
        if no_punct.islower():
            return mapped_word.lower()
        elif len(no_punct) > 1 and no_punct.isupper():
            return mapped_word.upper()
        elif len(no_punct) > 0 and no_punct[0].isupper():
            return mapped_word.capitalize()
        return mapped_word
    
    def add_punctuation_back(self, input_word, mapped_word, punct = ".,!?।\"”"):
        prefix_punct = ""
        suffix_punct = ""
        for char in input_word:
            if char in punct:
                prefix_punct += char
            else:
                break
        for char in reversed(input_word):
            if char in punct:
                suffix_punct = char + suffix_punct
            else:
                break
        mapped_word = prefix_punct + mapped_word + suffix_punct
        return mapped_word
        
    def denoise_functional(self, input):
        '''Replace functional words in sentence'''
        denoised_sentence = []
        for idx, input_word in enumerate(input.split()):
            no_punct, raw = self.preprocess(input_word)
            # if the word has an equivalent functional word in the high-resource language, replace it
            equivalent_hrl_word = self.get_top_equivalent(raw) or no_punct
            if self.is_hrl_word_functional(equivalent_hrl_word):
                mapped_word = equivalent_hrl_word
            else:
                mapped_word = no_punct
            mapped_word = self.maintain_casing(no_punct, mapped_word)
            mapped_word = self.add_punctuation_back(input_word, mapped_word)
            denoised_sentence.append(mapped_word)
        return " ".join(denoised_sentence)
    
    def denoise_content(self, input):
        '''Replace content words in sentence'''
        denoised_sentence = []
        for idx, input_word in enumerate(input.split()):
            no_punct, raw = self.preprocess(input_word)
            # assume that everything that is not a functional word is a content word
            equivalent_hrl_word = self.get_top_equivalent(raw) or no_punct
            if not self.is_hrl_word_functional(equivalent_hrl_word):
                mapped_word = equivalent_hrl_word
            else:
                mapped_word = no_punct
            mapped_word = self.maintain_casing(no_punct, mapped_word)
            mapped_word = self.add_punctuation_back(input_word, mapped_word)
            denoised_sentence.append(mapped_word)
        return " ".join(denoised_sentence)
    
    def denoise_all(self, input):
        '''Replace all words in sentence'''
        denoised_sentence = []
        for idx, input_word in enumerate(input.split()):
            no_punct, raw = self.preprocess(input_word)
            mapped_word = self.get_top_equivalent(raw) or no_punct
            mapped_word = self.maintain_casing(no_punct, mapped_word)
            mapped_word = self.add_punctuation_back(input_word, mapped_word)
            denoised_sentence.append(mapped_word)
        return " ".join(denoised_sentence)
        
    def report_changed_words(self, original_input, denoised_input):
        original_words = original_input.split()
        start_d = 0
        start_o = 0
        changes = 0
        for original_word in original_words:
            original_temp = original_word
            end_o = start_o + len(original_word)
            if end_o < len(original_input) and original_input[end_o] == " ":
                original_temp += " "
            start_o = end_o + 1
            if original_temp.lower() == denoised_input[start_d:start_d+len(original_temp)].lower():
                start_d += len(original_temp)
                while start_d < len(denoised_input) and denoised_input[start_d] == " ": # speed past extra spaces
                    start_d += 1
                continue  # Skip to the next word
            no_punct, raw = self.preprocess(original_word)
            mapped_words = self.get_all_equivalents(raw) or raw
            for mapped_word in mapped_words:
                _, potential_word = self.preprocess(mapped_word)
                potential_word = self.add_punctuation_back(original_word, potential_word)
                if " " in original_temp:
                    potential_word += " "
                if potential_word == denoised_input[start_d:start_d+len(potential_word)].lower().lstrip():
                    changes += 1
                    start_d += len(potential_word)
                    while start_d < len(denoised_input) and denoised_input[start_d] == " ": # speed past extra spaces
                        start_d += 1
                    break  # Exit inner loop if match is found
        return changes

# if __name__ == "__main__":
#     # SANITY CHECK
#     sentences = [
#         "“Avemu suggi di 4 misi ca annanzi èrunu diabbètici e ora no cchiù”, agghiuncìu iḍḍu.",
#         "Danius rissi, “Ora nun stamu facennu nènti. Chiamài e mannài email ô so collaburatùri cchiù prossimo e ricivìtti risposti amichevuli assài. Accom’ora chistu abbasta.”",
#         "Ring addinunziàu macari na società di sicurezza cuncurrenti, l’ADT Corporation."
#     ]
    
#     for sentence in sentences:
#         print("Raw Sentence")
#         print(sentence)
#         print("-" * 40)
#         for scheme in ["content", "functional", "all"]:
#             bilingual_lexicon_path = f"lexicons/scn-ita/scn-ita_{scheme}.json"
#             transducer = Denoiser('scn', 'ita', bilingual_lexicon_path)
#             denoised = transducer.denoise_all(sentence)
#             changed_words = transducer.report_changed_words(sentence, denoised)
#             print(f"Denoise {scheme} scheme")
#             print(denoised)
#             print(f"{changed_words} words changed")
#             print()
#         print()
