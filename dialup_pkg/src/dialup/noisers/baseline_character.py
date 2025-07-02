from .noise import Noise
import random
import os, sys
from .utils.misc import get_character_set, normalize_lang_codes

class RandomCharacterNoiserAugmenter(Noise):
    '''
    Noise type: switch out random characters with other characters
    Required params: lang, swap_theta
    Input format e.g.: {lang: hin, swap_theta: 0.1}
    '''
    def __init__(self, noise_params):
        '''Initialize noise with noise parameters
        Args:
            noise_params: dict, noise parameters, like {theta_1: 0.5}
        '''
        self.class_name = "RandomCharacterNoiser"
        self.required_keys = {"lang", "theta_random_char"}
        # self.required_keys = {"lang", "insert_theta", "delete_theta", "swap_theta"}
        self.check_noise_params(noise_params)

        
        self.lang = normalize_lang_codes(noise_params['lang'])
        # self.insert_theta = float(noise_params['insert_theta'])
        # self.delete_theta = float(noise_params['delete_theta'])
        self.theta_random_char = float(noise_params['theta_random_char'])

        # Initialize character set according to lang
        script, self.character_set = get_character_set(self.lang)
    
    def apply_noise(self, input):
        '''Apply noise to input
        Args:
            input: str, input text
        Returns:
            str, noised text
        '''
        # Apply noise
        # For each character, with probability swap_theta, 
        # swap it with another character from the same alphabet

        noised_input = ""
        for char in input:
            if char in {" ", "\n"}:
                noised_input += char
                continue
            if char in self.character_set:
                # If the character is in the character set, apply noise
                if random.random() < self.theta_random_char:
                    noised_input += random.choice(list(self.character_set - {char}))
                else:
                    noised_input += char
            else:
                # If not, leave the character alone
                noised_input += char
        return noised_input

    
    def find_posterior(text1, text2):
        '''Find the posterior MLE estimate of self.noise_params given text1 and text2
        Args:
            text1: str, text 1
            text2: str, text 2
        Returns:
            MLE estimate of self.noise_params (dict)
        '''
        raise NotImplementedError
