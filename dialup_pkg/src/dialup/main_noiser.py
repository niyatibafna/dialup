from .noisers.main import parse_noise_params, get_noisers, apply_noisers, apply_noisers_compose, apply_noisers_compose_augment, record_noiser_artifacts
from .noisers.utils.get_default_parameters import get_default_parameters

class Noiser:
    def __init__(self, noiser_params=None, lang=None, text_file=None):
        """Initialize the Noiser with noiser parameters."""
        if noiser_params is None:
            # We'll use default parameters if none are provided.
            noiser_params = get_default_parameters(lang, text_file)

        self.noiser_classes = get_noisers(noiser_params)
        self.noiser_params = noiser_params


    def apply_noise(self, input_text, verbose=False):
        """Apply noisers to the input text."""
        if verbose:
            print(f"Applying noisers with params: {self.noiser_params}")
        return apply_noisers_compose_augment(input_text, self.noiser_classes, verbose=verbose)

def print_languages_with_inbuilt_noising_support():
    """Returns a list of language pairs that have inbuilt support for noising."""
    supported_languages = [
    "hin",  # Hindi
    "ara",  # Arabic
    "ind",  # Indonesian
    "tur",  # Turkish
    "ita",  # Italian
    "hat",  # Haitian Creole
    "deu",  # German
    "eng",  # English
    "rus",  # Russian
    "spa",  # Spanish
    "fra",  # French
    ]
    print("Supported languages for artificial related dialect / variant generation: ", supported_languages)
    print("For any other language, you can include support by following the steps here: https://github.com/niyatibafna/dialup/tree/master/mtd/generating_artificial_dialects.")

# Replace the below noise_params with the your noise parameters.
# text_file needs to point to a raw text file for the language.
## It's used for training char-gram models and suffix identification.

# --------------------------------------------

# Example run

# all_noise_params = {
#   "lexical_aug": {
#     "lang": "hi",
#     "theta_content_global": 0.001,
#     "theta_func_global": 0.8,
#     "chargram_length": 3,
#     "text_file": "/export/b08/nbafna1/data/wikimatrix/en-hi/WikiMatrix.en-hi.hi"
#   },
#   "morph_aug": {
#     "lang": "hi",
#     "theta_morph_global": 0.3,
#     "text_file": "/export/b08/nbafna1/data/wikimatrix/en-hi/WikiMatrix.en-hi.hi"
#   },
#   "phonological_aug": {
#     "lang": "hi",
#     "theta_phon": 0.07,
#     "text_file": "/export/b08/nbafna1/data/wikimatrix/en-hi/WikiMatrix.en-hi.hi"
#   },
#   "random_char_aug": {
#     "lang": "hi",
#     "theta_random_char": 0.0
#   },
#   "random_word_aug": {
#     "lang": "hi",
#     "theta_random_word": 0.0,
#     "text_file": "/export/b08/nbafna1/data/wikimatrix/en-hi/WikiMatrix.en-hi.hi"
#   },
#   "semantic_aug": {
#     "lang": "hi",
#     "theta_semantic_global": 0.2,
#     "text_file": "/export/b08/nbafna1/data/wikimatrix/en-hi/WikiMatrix.en-hi.hi"
#   }
# }


# print(f"Noise Parameters: {all_noise_params}")

# noiser_classes = get_noisers(all_noise_params)
# print(f"Noiser Classes: {noiser_classes}")

# inputs = [
#     "रवि रोज़ सुबह जल्दी उठता था।",
#     "एक दिन उसने देखा कि एक कुत्ता उसके दरवाजे पर बैठा था।",
#     "रवि ने उसे दूध दिया और कुत्ता ख़ुश हो गया।",
#     "अगले दिन भी कुत्ता वापस आ गया और रवि उसका दोस्त बन गया।",
#     "अब वह कुत्ता हर दिन रवि के साथ खेलने आता है।"
# ]

# for i in range(5):
#     # We apply 5 different augmentations to the same input using the above noiser config.
#     print(f"Augmentation: {i}")
#     for idx, input in enumerate(inputs):
#         # noised = apply_noisers(input, noiser_classes, verbose=True)
#         noised = apply_noisers_compose_augment(input, noiser_classes, verbose=False)
#         print(f"Input: {idx}")
#         print(f"Input: {input.strip()}")
#         print(f"Noised: {noised.strip()}")

