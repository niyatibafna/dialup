from importlib.resources import files as pkg_files
from .denoisers.main import Denoise
import json
import os

class Denoiser:
    def __init__(self, lrl, hrl, strategy = "functional", bilingual_lexicon_path=None):
        """
        Strategy can be 'functional', 'content', or 'all'.
        """
        if strategy not in ["functional", "content", "all"]:
            raise ValueError("Strategy must be one of 'functional', 'content', or 'all'.")
        if strategy != "functional" and bilingual_lexicon_path is None:
            raise ValueError("Bilingual lexicon path must be provided for 'content' or 'all' strategies.")

        bilingual_lexicon = self.get_bilingual_lexicon(lrl, hrl, bilingual_lexicon_path)

        self.strategy = strategy
        self.denoiser = Denoise(lrl, hrl, bilingual_lexicon)

    
    def get_bilingual_lexicon(self, lrl, hrl, path=None):
        '''Get bilingual lexicon that translate from low to high-resource languages from the file'''
        if not path:
    
            # Default path in the package
            path = pkg_files('dialup').joinpath('data', 'lexicons', f"{lrl}-{hrl}", f"{lrl}-{hrl}_functional.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Language pair {lrl}-{hrl} not supported.")

        with open(path) as f:
            return json.load(f)
        

    def denoise(self, input_text, verbose=False):
        """Denoise the input text."""
        if verbose:
            print(f"Denoising with strategy: {self.strategy}")
        
        strategy_to_func = {
            "functional": self.denoiser.denoise_functional,
            "content": self.denoiser.denoise_content,
            "all": self.denoiser.denoise_all
        }
        denoise_func = strategy_to_func[self.strategy]
        return denoise_func(input_text)


def print_language_pairs_with_inbuilt_denoising_support():
    """Returns a list of language pairs that have inbuilt support."""
    supported_language_pairs = ['acf-hat', 'ajp-arb', 'arz-arb', 'bho-hin', 'crs-hat', 'glg-ita', 'jav-ind', 'mai-hin', 'pag-ind', 'scn-ita', 'sun-ind', 'vec-ita', 'acm-arb', 'apc-arb', 'ast-ita', 'cat-ita', 'fij-ind', 'hne-hin', 'lij-ita', 'mfe-hat', 'plt-ind', 'smo-ind', 'tgl-ind', 'zsm-ind', 'acq-arb', 'ars-arb', 'awa-hin', 'ceb-ind', 'fra-ita', 'ilo-ind', 'lmo-ita', 'mri-ind', 'por-ita', 'spa-ita', 'tuk-tur', 'aeb-arb', 'ary-arb', 'azj-tur', 'crh-tur', 'fur-ita', 'ita-ita', 'mag-hin', 'oci-ita', 'ron-ita', 'srd-ita', 'uzn-tur']
    print("Language pairs with included function word lexicons (strategy='functional'): ", 
    supported_language_pairs)
    support_hrls = ['hin', 'ara', 'ind', 'tur', 'ita', 'hat', 'deu', 'eng', 'rus', 'spa', 'fra']
    print(f"You can also perform denoising for any of the following high-resource languages: {support_hrls} with any other language provided you pass a bilingual lexicon.")
    print("For any other language pair or strategy, please provide the file path to a bilingual lexicon.")