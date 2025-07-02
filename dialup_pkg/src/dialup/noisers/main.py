from .phonological import PhonologicalNoiserAugmenter
from .baseline_character import RandomCharacterNoiserAugmenter
from .baseline_word import RandomWordNoiserAugmenter
from .lexical import LexicalNoiserAugmenter
from .morphological import MorphologicalNoiserAugmenter
# from .semantic import SemanticNoiserAugmenter
# from google_translate import GoogleTranslateNoiser

from collections import defaultdict
import regex

NOISE_REGISTRY = {
    # Not releasing SemanticNoiserAugmenter as part of the package yet
    'phonological_aug': PhonologicalNoiserAugmenter,
    'lexical_aug': LexicalNoiserAugmenter,
    'morph_aug': MorphologicalNoiserAugmenter,
    'random_char_aug': RandomCharacterNoiserAugmenter,
    'random_word_aug': RandomWordNoiserAugmenter,
    # 'semantic_aug': SemanticNoiserAugmenter,
}

def load_noise_params_from_file(noise_params_file):
    '''
    Load noise parameters from JSON file, formatted as follows:
    {
        "phonological_aug": {
            "theta_phon": 0.5,
            "lang": "hin",
            ...
        },
        "lexical_aug": {
            "theta_content": 0.3,
            ...
        }
    }
    '''
    import json
    with open(noise_params_file, "r") as f:
        noise_params = json.load(f)
    return noise_params

def parse_noise_params(noise_params_str):
    '''
    Parse noise parameters e.g. phonological-theta_1=0.5,theta_2=0.2;syntax-theta_2=0.5
    Args:
        noise_params_str: str, noise parameters
    Returns:
        dict, noise parameters, like {phonological: {theta_1: 0.5}}
    
    '''
    if noise_params_str == "":
        return defaultdict(dict)

    all_noise_params = defaultdict(dict)

    # We will ignore anything enclosed in <>
    ## Extract text enclosed in <>
    text_files = regex.findall(r'<(.*?)>', noise_params_str)
    ## Replace text enclosed in <> with a placeholder
    noise_params_str = regex.sub(r'<(.*?)>', "<placeholder>", noise_params_str)

    # print(noise_params_str)

    all_noise_type_params = noise_params_str.split(";")
    for noise_type_params in all_noise_type_params: # phonological-theta_1=0.5,theta_2=0.2
        noise_type = noise_type_params.split("-")[0] # phonological
        noise_type_params = noise_type_params.split("-")[1] # theta_1=0.5,theta_2=0.2
        noise_type_params = noise_type_params.split(",") # [theta_1=0.5,theta_2=0.2]
        for noise_param in noise_type_params:
            param_name = noise_param.split("=")[0]
            param_value = noise_param.split("=")[1]
            print(param_name, param_value)
            # If param_value is a placeholder, replace it with the actual text_file
            if param_value == "<placeholder>":
                all_noise_params[noise_type][param_name] = text_files.pop(0)
            else:
                try:
                    all_noise_params[noise_type][param_name] = float(param_value)
                except:
                    all_noise_params[noise_type][param_name] = param_value

    return all_noise_params
    

def get_noisers(all_noise_params):
    '''Initialize noisers with noise parameters
    Args:
        input: str, input text
        all_noise_params: dict, noise parameters, like {phonological: {theta_1: 0.5}}
    Returns:
        noise_classes: list, list of noiser class objects
    '''
    if not all_noise_params:
        return list()

    # Initialize noiser objects from noise type classes
    noise_classes = list()
    for noise_type, noise_params in all_noise_params.items():
        theta_params = {k: v for k, v in noise_params.items() if "theta" in k}
        if all([theta_value == 0 for theta_value in theta_params.values()]):
            print(f"Skipping {noise_type} as all thetas are 0")
            continue
        noiser = NOISE_REGISTRY[noise_type](noise_params)
        noise_classes.append(noiser)
    
    return noise_classes

def apply_noisers(input, noise_classes, verbose = False):
    '''Apply noise to input
    Args:
        input: str, input text
        noise_classes: list, list of noisers
    Returns:
        str, noised text
    '''
    if len(noise_classes) > 1:
        return apply_noisers_compose(input, noise_classes, verbose)

    for noiser in noise_classes:
        if verbose:
            print(f"Applying noise: {noiser}")
        input = noiser.apply_noise(input)
    return input

def apply_noisers_compose(input, noise_classes, verbose = False):
    '''Apply noise to input, compose all noisers
    Args:
        input: str, input text
        noise_classes: list, list of noisers
    Returns:
        str, noised text
    '''
    noise_type_output = dict()
    for noiser in noise_classes:
        if verbose:
            print(f"Applying noise: {noiser}")
        noise_type_output[noiser.class_name] = noiser.apply_noise(input).split()

        try:
            assert len(input.split()) == len(noise_type_output[noiser.class_name])
        except:
            print("WARNING: Lengths of input and noised output do not match")
            print(f"Input: {input}")
            print(f"Noised Output: {noise_type_output[noiser.class_name]}")
            print(f"Lengths: {len(input.split())}, {len(noise_type_output[noiser.class_name])}")
            print(noiser.class_name)
            noise_type_output[noiser.class_name] = input.split()

    # print(noise_type_output)
    
    # If some kind of noise is not applied, we will add it as the original input
    for noiser in {"GlobalPhonologicalNoiser", "GlobalLexicalNoiser", "GlobalMorphologicalNoiser"}:
        if noiser not in noise_type_output:
            noise_type_output[noiser] = input.split()

    # Now we will compose these outputs
    ## We assume that the order is phonological, morphological, lexical
    final_output = list()

    for i, word in enumerate(input.split()):
        noised_word = word
        if noise_type_output['GlobalLexicalNoiser'][i] != word:
            # If lexical noiser has changed the word, we will use that
            noised_word = noise_type_output['GlobalLexicalNoiser'][i]
            final_output.append(noised_word)
            continue
        # Else we take the phonological noised word
        noised_word = noise_type_output['GlobalPhonologicalNoiser'][i]

        # If morphological change changed the suffix, we'll apply the new suffix
        if noise_type_output['GlobalMorphologicalNoiser'][i] != word:
            morph_noised_word = noise_type_output['GlobalMorphologicalNoiser'][i]
            ## First let's find the suffix
            ## The stem is simply the longest shared prefix
            stem = ""
            for j in range(min(len(word), len(morph_noised_word))):
                if word[j] == morph_noised_word[j]:
                    stem += word[j]
                else:
                    break
            ## The suffix is the remaining part
            morph_noised_suffix = morph_noised_word[len(stem):]

            phon_noised_stem = noised_word[:len(stem)]
            noised_word = phon_noised_stem + morph_noised_suffix

            ## All of the above assumes that phon noise preserves the length of the word
            ## which thankfully it does
        final_output.append(noised_word)

    return " ".join(final_output)

def apply_noisers_compose_augment(input, noise_classes, verbose = False):
    '''Apply noise to input, compose all noisers
    Args:
        input: str, input text
        noise_classes: list, list of noisers
    Returns:
        str, noised text
    '''

    # If the SemanticNoiserAugmenter is present, we will apply it first, before any other noisers
    ## This is because the semantic noiser models a different lexical choice for the word at some point
    ## in an ancestor language, and the other noisers model the changes that have happened since then

    noise_classes = noise_classes.copy()

    for noiser in noise_classes:
        if noiser.class_name == "SemanticNoiserAugmenter":
            try:
                input = noiser.apply_noise(input)
            except:
                print(f"Error in applying {noiser}")
                print(f"Input: {input}")
            noise_classes.remove(noiser)
            break


    noise_type_output = dict()
    for noiser in noise_classes:
        if verbose:
            print(f"Applying noise: {noiser.class_name}")
        noise_type_output[noiser.class_name] = noiser.apply_noise(input).split()
        if verbose:
            print(f"Output: {noise_type_output[noiser.class_name]}")

        try:
            assert len(input.split()) == len(noise_type_output[noiser.class_name])
        except:
            print("WARNING: Lengths of input and noised output do not match")
            print(f"Input: {input}")
            print(f"Noised Output: {noise_type_output[noiser.class_name]}")
            print(f"Lengths: {len(input.split())}, {len(noise_type_output[noiser.class_name])}")
            print(noiser.class_name)
            noise_type_output[noiser.class_name] = input.split()

    if verbose:
        print("Composing all noisers...")
    
    # If some kind of noise is not applied, we will add it as the original input
    for noiser in {"PhonologicalNoiserAugmenter", "LexicalNoiserAugmenter", "MorphologicalNoiserAugmenter", "RandomCharacterNoiserAugmenter", "RandomWordNoiserAugmenter"}:
        if noiser not in noise_type_output:
            noise_type_output[noiser] = input.split()

    # Now we will compose these outputs
    ## We assume that the order is phonological, morphological, lexical
    ## !!! Note: We currently assume that either the random lexical noiser or the linguistic lexical noiser
    ## are applied, and don't bother about composing these.
    ## Similarly, we assume that either the random character noiser or the phonological noiser are applied, 
    ## so the order of these two doesn't matter.

    final_output = list()

    for i, word in enumerate(input.split()):
        noised_word = word

        # If lexical noiser has changed the word, we will use that
        if noise_type_output['LexicalNoiserAugmenter'][i] != word:
            noised_word = noise_type_output['LexicalNoiserAugmenter'][i]
            final_output.append(noised_word)
            continue

        # If random lexical noiser has changed the word, we will use that
        if noise_type_output['RandomWordNoiserAugmenter'][i] != word:
            noised_word = noise_type_output['RandomWordNoiserAugmenter'][i]
            final_output.append(noised_word)
            continue

        # If random character noiser has changed the word, we will use that
        noised_word = noise_type_output['RandomCharacterNoiserAugmenter'][i]

        # Else we take the phonological noised word
        noised_word = noise_type_output['PhonologicalNoiserAugmenter'][i]

        # If morphological change changed the suffix, we'll apply the new suffix
        if noise_type_output['MorphologicalNoiserAugmenter'][i] != word:
            morph_noised_word = noise_type_output['MorphologicalNoiserAugmenter'][i]
            ## First let's find the suffix
            ## The stem is simply the longest shared prefix
            stem = ""
            for j in range(min(len(word), len(morph_noised_word))):
                if word[j] == morph_noised_word[j]:
                    stem += word[j]
                else:
                    break
            ## The suffix is the remaining part
            morph_noised_suffix = morph_noised_word[len(stem):]

            phon_noised_stem = noised_word[:len(stem)]
            noised_word = phon_noised_stem + morph_noised_suffix

            ## All of the above assumes that phon noise preserves the length of the word
            ## which thankfully it does

        final_output.append(noised_word)

    return " ".join(final_output)


def record_noiser_artifacts(noise_classes):
    '''Save noiser artifacts to output file
    Args:
        noise_classes: list, list of noisers
    '''
    for noiser in noise_classes:
        noiser.record_noiser_artifacts()