from collections import defaultdict

def normalize_lang_codes(lang):
    '''Make everything ISO 639-3'''
    # {Spanish: es, Hindi: hi, Russian: ru, Indonesian: id, English: en
# Catalan: ca, Galician: gl, Portuguese: pt, Bengali: bn, Marathi: mr, Nepali: ne, Slovak: sk, Serbian: sr
# Croatian: hr, Ukrainian: uk
# Bulgarian, Belarusian
# }
    lang = lang.lower()
    to_iso3 = {
        "esp": "spa",
        "en": "eng",
        "de": "deu",
        "hi": "hin",
        "ar": "arb",
        "ru": "rus",
        "es": "spa",
        "fr": "fra",
        "cs": "ces",
        "id": "ind",
        "ca": "cat",
        "gl": "glg",
        "pt": "por",
        "bn": "ben",
        "mr": "mar",
        "ne": "nep",
        "sk": "slk",
        "sr": "srp",
        "hr": "hrv",
        "uk": "ukr",
        "bg": "bul",
        "be": "bel",
        "tr": "tur",
        "it": "ita",
        "ht": "hat",
    }
    return to_iso3.get(lang, lang)


def get_lang_name(lang_code):
    '''Get language name from language code
    Args:
        lang_code: str, language code
    Returns:
        str, language name
    '''
    lang_code = normalize_lang_codes(lang_code)
    lang_map = {
        "eng": "English",
        "deu": "German",
        "hin": "Hindi",
        "arb": "Arabic",
        "rus": "Russian",
        "spa": "Spanish",
        "ind": "Indonesian",
        "cat": "Catalan",
        "glg": "Galician",
        "por": "Portuguese",
        "ben": "Bengali",
        "mar": "Marathi",
        "nep": "Nepali",
        "slk": "Slovak",
        "srp": "Serbian",
        "hrv": "Croatian",
        "ukr": "Ukrainian",
        "bho": "Bhojpuri",
        "bel": "Belarusian",
        "bos": "Bosnian",
        "bul": "Bulgarian",
        "ces": "Czech",
        "fra": "French",
        "tur": "Turkish",
        "ita": "Italian",
        "hat": "Haitian Creole",
    }
    if lang_code not in lang_map:
        print(f"WARNING: Language code {lang_code} not found in lang_map")
    return lang_map.get(lang_code, lang_code)

def related_to_lang(lang):
    '''
    Given a language, it returns the high-resource language related to it.
    '''
    lang = normalize_lang_codes(lang)
    # "bho": "bho_Deva",
    # "awa": "awa_Deva",
    # "mag": "mag_Deva",
    # "mai": "mai_Deva",
    # "hne": "hne_Deva",
    # "zsm": "zsm_Latn",
    # "oci": "oci_Latn",
    # "glg": "glg_Latn",
    # "dan": "dan_Latn",
    # "nor": "nor_Latn",
    # "isl": "isl_Latn",
    # "swe": "swe_Latn",
    # "acm": "acm_Arab",
    # "acq": "acq_Arab",
    # "aeb": "aeb_Arab",
    # "ajp": "ajp_Arab",
    # "apc": "apc_Arab",
    # "ars": "ars_Arab",
    # "ary": "ary_Arab",
    # "arz": "arz_Arab",
    related_lrls = {
        "hin": {"bho", "awa", "mag", "mai", "hne"},
        "ind": {"zsm"},
        "spa": {"glg"},
        "fra": {"oci"},
        "deu": {"dan", "nor", "isl", "swe"},
        "arb": {"acm", "acq", "aeb", "ajp", "apc", "ars", "ary", "arz"},
    }

    for hrl in related_lrls:
        if lang in related_lrls[hrl]:
            return hrl


def identify_script(text):
    '''Identify script of text
    Args:
        text: str, input text
    Returns:
        str, script of text
    '''
    # Identify script of text
    char_set = {
        "latin": set(chr(i) for i in range(65, 91)) | set(chr(i) for i in range(97, 123)),
        "devanagari": set(chr(i) for i in range(2304, 2432)),
        "arabic": set(chr(i) for i in range(1536, 1792)),
        "cyrillic": set(chr(i) for i in range(1024, 1280)),
    }
    # We'll find a majority script
    script_counts = {script: 0 for script in char_set.keys()}
    for char in text:
        for script, char_set in char_set.items():
            if char in char_set:
                script_counts[script] += 1

    return max(script_counts, key=script_counts.get)

def ipa_char_maps():
    '''
    Defines the maps ipa2chars and char2ipas for a given language (assuming one script per language).
    Returns:
        ipa_to_script_chars: dict, mapping from IPA to script characters for each language, {lang: {ipa: set(script_chars)}}
        script_to_ipa_chars: dict, mapping from script characters to IPA for each language: {lang: {script_char: set(ipas)}}
        
    '''
    # script, char_set = get_character_set(lang)
    # lang = normalize_lang_codes(lang)

    eng_char2ipas = {
    'a': {'æ', 'ɑ'},
    'b': {'b'},
    'c': {'k', 's'},
    'd': {'d'},
    'e': {'ɛ', 'i', 'e'},
    'f': {'f'},
    'g': {'ɡ', 'dʒ'},
    'h': {'h'},
    'i': {'ɪ', 'i', 'aɪ'},
    'j': {'dʒ'},
    'k': {'k'},
    'l': {'l'},
    'm': {'m'},
    'n': {'n'},
    'o': {'ɑ', 'ʌ', 'o'},
    'p': {'p'},
    'q': {'k'},
    'r': {'r'},
    's': {'s'},
    't': {'t'},
    'u': {'ʊ', 'u'},
    'v': {'v'},
    'w': {'w'},
    'x': {'ks'},
    'y': {'j', 'i'},
    'z': {'z'}
    }
    
    latin_to_ipa_set_haitian = {
        'p': {'p'},
        'b':{'b'},
        't': {'t'},
        'd': {'d'},
        'k': {'k'},
        'g': {'ɡ'},
        'm': {'m'},
        'n': {'n'},
        'ng': {'ŋ'},
        'f': {'f'},
        'v': {'v'},
        's': {'s'},
        'z': {'z'},
        'ch': {'ʃ'},
        'j': {'ʒ'},
        'r': {'ɣ'},
        'h': {'h'},
        'l': {'l'},
        'w': {'w'},
        'y': {'j'},
        'a': {'a', 'ɑ'},
        'à': {'a', 'ɑ'},
        'an': {'ã', 'ɑ̃'},
        'e': {'e'},
        'è': {'ɛ'},
        'en': {'ɛ̃'},
        'i': {'i'},
        'o': {'o'},
        'ò': {'ɔ'},
        'on': {'ɔ̃'},
        'ou': {'u'},
        'ui': {'ɥi'},
    }

    devanagari_to_ipa_set = {
        'अ': {'ʌ', 'ə'},
        'आ': {'aː'},
        'इ': {'ɪ'},
        'ई': {'iː'},
        'उ': {'ʊ'},
        'ऊ': {'uː'},
        'ऋ': {'r̩', 'rɪ'},
        'ए': {'eː'},
        'ऐ': {'aɪ'},
        'ओ': {'oː'},
        'औ': {'aʊ'},
        'अं': {'əŋ'},
        'अः': {'əh'},
        'क': {'k'},
        'ख': {'kʰ'},
        'ग': {'ɡ'},
        'घ': {'ɡʱ'},
        'ङ': {'ŋ'},
        'च': {'t͡ʃ'},
        'छ': {'t͡ʃʰ'},
        'ज': {'d͡ʒ'},
        'झ': {'d͡ʒʱ'},
        'ञ': {'ɲ'},
        'ट': {'ʈ'},
        'ठ': {'ʈʰ'},
        'ड': {'ɖ'},
        'ढ': {'ɖʱ'},
        'ण': {'ɳ'},
        'त': {'t̪'},
        'थ': {'t̪ʰ'},
        'द': {'d̪'},
        'ध': {'d̪ʱ'},
        'न': {'n'},
        'प': {'p'},
        'फ': {'pʰ'},
        'ब': {'b'},
        'भ': {'bʱ'},
        'म': {'m'},
        'य': {'j'},
        'र': {'r'},
        'ल': {'l'},
        'व': {'v'},
        'श': {'ʃ'},
        'ष': {'ʂ'},
        'स': {'s'},
        'ह': {'h'},
        'ळ': {'ɭ'},
        'क्ष': {'kʂ'},
        'ज्ञ': {'d͡ʒɲ'},
        'ा': {'aː'},
        'ि': {'ɪ'},
        'ी': {'iː'},
        'ु': {'ʊ'},
        'ू': {'uː'},
        'ृ': {'r̩', 'rɪ'},
        'े': {'eː'},
        'ै': {'aɪ'},
        'ो': {'oː'},
        'ौ': {'aʊ'},
    }

    latin_to_ipa_set_indonesian = {
        'a': {'a'},
        'b': {'b'},
        'c': {'t͡ʃ'},
        'd': {'d'},
        'e': {'ə', 'e'},
        'f': {'f'},
        'g': {'ɡ'},
        'h': {'h'},
        'i': {'i'},
        'j': {'d͡ʒ'},
        'k': {'k'},
        'l': {'l'},
        'm': {'m'},
        'n': {'n'},
        'o': {'o'},
        'p': {'p'},
        'q': {'k'},
        'r': {'r'},
        's': {'s'},
        't': {'t'},
        'u': {'u'},
        'v': {'v'},
        'w': {'w'},
        'x': {'ks'},
        'y': {'j'},
        'z': {'z'},
    }

    latin_to_ipa_set_spanish = {
        'a': {'a'},
        'b': {'b'},
        'c': {'k', 's', 'θ'},
        'd': {'d'},
        'e': {'e'},
        'f': {'f'},
        'g': {'ɡ', 'x'},
        'h': {''},  # Silent in most cases
        'i': {'i'},
        'j': {'x', 'ʝ', 'h'},  # Variations depending on region and word
        'k': {'k'},
        'l': {'l'},
        'm': {'m'},
        'n': {'n'},
        'o': {'o'},
        'p': {'p'},
        'q': {'k'},
        'r': {'r', 'ɾ'},  # Rolled 'r' and flap 'r'
        's': {'s'},
        't': {'t'},
        'u': {'u'},
        'v': {'b', 'β'},
        'w': {'w'},
        'x': {'ks', 'ɡz'},  # In words of foreign origin
        'y': {'ʝ', 'j', 'i'},  # Variations depending on region and word
        'z': {'s', 'z'}
    }

    latin_to_ipa_set_french = {
    'a': {'a', 'ɑ'},
    'b': {'b'},
    'c': {'k', 's'},
    'd': {'d'},
    'e': {'ə', 'e', 'ɛ'},
    'f': {'f'},
    'g': {'ɡ'},
    'h': {''},  # In French, 'h' is mostly silent
    'i': {'i', 'j'},
    'j': {'ʒ'}, 
    'k': {'k'},
    'l': {'l'},
    'm': {'m'},
    'n': {'n'},
    'o': {'o', 'ɔ'},
    'p': {'p'},
    'q': {'k'},
    'r': {'ʁ', 'ʀ', 'ʀ̥'},  # The French 'r' can vary in pronunciation
    's': {'s', 'z'},
    't': {'t'},
    'u': {'y', 'u'}, 
    'v': {'v'},
    'w': {'w'},
    'x': {'ks'},  # Rare in French
    'y': {'i', 'j', 'ɥ'},
    'z': {'z'},
    'à': {'a'},
    'â': {'ɑ'},  # In some French accents
    'ç': {'s'},  # Before 'a', 'o', or 'u'
    'è': {'ɛ'},
    'é': {'e'},
    'ê': {'ɛ'},  # In some French accents
    'ë': {'ɛ'},  # In some French accents
    'î': {'i'},  # In some French accents
    'ï': {'i'},  # In some French accents
    'ô': {'o'},  # In some French accents
    'ù': {'y', 'u'},  # In some French accents
    'û': {'y', 'u'},  # In some French accents
    'ü': {'y'},  # In some French accents
    'ÿ': {'i'},  # In some French accents
    }

    cyrillic_to_ipa_set_russian = {
        'а': {'a'},
        'б': {'b'},
        'в': {'v'},
        'г': {'ɡ'},
        'д': {'d'},
        'е': {'je', 'e', 'ɛ'},  # 'e' can represent both /je/ and /e/ depending on stress
        'ё': {'jo', 'o', 'ɔ'},  # 'o' can represent both /jo/ and /o/ depending on stress
        'ж': {'ʐ'},
        'з': {'z'},
        'и': {'i'},
        'й': {'j'},  # Semi-vowel /j/
        'к': {'k'},
        'л': {'l'},
        'м': {'m'},
        'н': {'n'},
        'о': {'o'},
        'п': {'p'},
        'р': {'r'},
        'с': {'s'},
        'т': {'t'},
        'у': {'u'},
        'ф': {'f'},
        'х': {'x', 'h'},  # Variations depending on dialect
        'ц': {'ts'},
        'ч': {'tɕ'},
        'ш': {'ʂ'},
        'щ': {'ɕtɕ'},
        'ъ': {''},  # Hard sign, often not pronounced
        'ы': {'ɨ'},
        'ь': {''},  # Soft sign, affects the preceding consonant but doesn't add a sound
        'э': {'ɛ'},
        'ю': {'ju', 'u', 'ʉ'},  # 'u' can represent both /ju/ and /u/ depending on stress
        'я': {'ja', 'a', 'æ'},  # 'a' can represent both /ja/ and /a/ depending on stress
    }

    arabic_to_ipa_set_msa = {
        'ا': {'ʔ', 'aː'},  # Alif
        'ب': {'b'},         # Ba
        'ت': {'t'},         # Ta
        'ث': {'θ'},         # Tha
        'ج': {'dʒ'},        # Jim
        'ح': {'ħ'},         # Hha
        'خ': {'χ'},         # Kha
        'د': {'d'},         # Dal
        'ذ': {'ð'},         # Thal
        'ر': {'r'},         # Ra
        'ز': {'z'},         # Zain
        'س': {'s'},         # Sin
        'ش': {'ʃ'},         # Shin
        'ص': {'sˤ'},        # Sad
        'ض': {'dˤ'},        # Dad
        'ط': {'tˤ'},        # Ta
        'ظ': {'ðˤ'},        # Za
        'ع': {'ʕ'},         # 'Ayn
        'غ': {'ɣ'},         # Ghayn
        'ف': {'f'},         # Fa
        'ق': {'q'},         # Qaf
        'ك': {'k'},         # Kaf
        'ل': {'l'},         # Lam
        'م': {'m'},         # Mim
        'ن': {'n'},         # Nun
        'ه': {'h'},         # Ha
        'و': {'w', 'uː'},   # Waw
        'ي': {'j', 'iː'},   # Ya
        'ة': {'aː'},        # Ta marbuta
        'ى': {'aː'},        # Alif maksura
        'ء': {'ʔ'},          # Hamza
        # "'َ'": {'a'}, # fatHa
        # "'ِ'": {'i'}, # kasra
        # "'ٌ'": {'un'}, # Damma tanwiin
        # "'ٍ'": {'in'}, # kasra tanwiin
        # "'ً'": {'n'}, # tanwiin
        # "'ّ'": {'ː'}, # shadda (this is for gemination. Not sure how you'd represent this) 

    }

    # Resample if not in alphabet

    latin_to_ipa_set_german = {
    'a': {'a', 'aː'},
    'b': {'b'},
    'c': {'k', 't͡s'},  # 'c' can represent both /k/ and /t͡s/
    'd': {'d'},
    'e': {'e', 'ɛ', 'eː'},
    'f': {'f'},
    'g': {'ɡ'},
    'h': {'h'},
    'i': {'i', 'ɪ'},
    'j': {'j'},
    'k': {'k'},
    'l': {'l'},
    'm': {'m'},
    'n': {'n'},
    'o': {'o', 'ɔ', 'oː'},
    'p': {'p'},
    'q': {'k', 'kv'},  # 'q' typically represents /kv/ in German
    'r': {'r', 'ʁ'},  # 'r' can represent both the alveolar tap /r/ and the uvular fricative /ʁ/
    's': {'z', 's', 'ʃ'},  # 's' can represent /z/, /s/, and /ʃ/ depending on position and word
    't': {'t'},
    'u': {'u', 'ʊ', 'uː'},
    'v': {'f', 'v'},  # 'v' can represent both /f/ and /v/
    'w': {'v', 'w'},  # 'w' can represent both /v/ and /w/
    'x': {'ks'},       # 'x' represents /ks/
    'y': {'y', 'ʏ', 'iː'},  # 'y' can represent /y/, /ʏ/, and /iː/
    'z': {'ts', 's', 'ʦ'},  # 'z' can represent /ts/, /s/, and /ʦ/
    'ä': {'ɛ', 'eː'},  # 'ä' can represent both /ɛ/ and /eː/
    'ö': {'ø', 'øː'},  # 'ö' can represent both /ø/ and /øː/
    'ü': {'y', 'yː'},  # 'ü' can represent both /y/ and /yː/
    'ß': {'s'}         # 'ß' represents /s/ or /sː/
    }

    latin_to_ipa_set_turkish = {
    'a': {'a'},
    'b': {'b'},
    'c': {'d͡ʒ'},  # "c" in Turkish is like English "j" in "judge"
    'ç': {'t͡ʃ'},  # "ç" is pronounced as "ch" in "chair"
    'd': {'d'},
    'e': {'e', 'ɛ'},  # Depending on the word, "e" can be close-mid or open-mid
    'f': {'f'},
    'g': {'ɡ'},
    'ğ': {'ɰ'},  # "ğ" (yumuşak ge) is a gliding sound, often lengthening the preceding vowel
    'h': {'h'},
    'ı': {'ɯ'},  # "ı" is a close back unrounded vowel
    'i': {'i'},  # "i" is a close front unrounded vowel
    'j': {'ʒ'},  # "j" in Turkish is like the "s" in "measure"
    'k': {'k'},
    'l': {'l'},  # Typically a clear "l", sometimes velarized depending on context
    'm': {'m'},
    'n': {'n'},
    'o': {'o'},  # "o" is a close-mid back rounded vowel
    'ö': {'ø'},  # "ö" is a front rounded vowel, like in German or French
    'p': {'p'},
    'r': {'ɾ'},  # "r" is a tap, but can be pronounced with a slight trill
    's': {'s'},
    'ş': {'ʃ'},  # "ş" is like "sh" in "shoe"
    't': {'t'},
    'u': {'u'},  # "u" is a close back rounded vowel
    'ü': {'y'},  # "ü" is a close front rounded vowel
    'v': {'v'},
    'y': {'j'},  # "y" is a palatal glide, as in "yes"
    'z': {'z'}
    }

    latin_to_ipa_set_italian = {
    'a': {'a'},  # Open front vowel
    'b': {'b'},  # Voiced bilabial plosive
    'c': {'k', 't͡ʃ'},  # "k" before a/o/u, "t͡ʃ" before e/i (like in "ciao")
    'd': {'d'},  # Voiced dental plosive
    'e': {'e', 'ɛ'},  # Close-mid front unrounded vowel, open-mid front unrounded vowel
    'f': {'f'},  # Voiceless labiodental fricative
    'g': {'ɡ', 'd͡ʒ'},  # "ɡ" before a/o/u, "d͡ʒ" before e/i (like in "gelato")
    'h': {'h'},  # Only in foreign words, rarely pronounced
    'i': {'i', 'j'},  # Close front unrounded vowel, or glide "j" in diphthongs (like in "piano")
    'l': {'l'},  # Voiced alveolar lateral approximant
    'm': {'m'},  # Voiced bilabial nasal
    'n': {'n', 'ŋ'},  # "n" is alveolar, but becomes "ŋ" before velar consonants like "g" or "k"
    'o': {'o', 'ɔ'},  # Close-mid back rounded vowel, open-mid back rounded vowel
    'p': {'p'},  # Voiceless bilabial plosive
    'q': {'k'},  # Always followed by "u" in "qu", pronounced "kw"
    'r': {'r'},  # Voiced alveolar trill
    's': {'s', 'z'},  # Voiceless alveolar fricative, voiced in some contexts like "rosa"
    't': {'t'},  # Voiceless dental plosive
    'u': {'u', 'w'},  # Close back rounded vowel, or glide "w" in diphthongs (like in "uomo")
    'v': {'v'},  # Voiced labiodental fricative
    'z': {'t͡s', 'd͡z'},  # Voiceless and voiced affricates, like "pizza" (t͡s) or "zebra" (d͡z)
    'j': {'j'},  # Only in some loanwords, pronounced as "y" sound (like "yoga")
    'x': {'ks'},  # Used in some loanwords, pronounced as "ks" (like "extra")
    }




    script_to_ipa_chars = {
        'eng': eng_char2ipas,
        'hin': devanagari_to_ipa_set,
        'ind': latin_to_ipa_set_indonesian,
        'hat': latin_to_ipa_set_haitian,
        'spa': latin_to_ipa_set_spanish,
        'rus': cyrillic_to_ipa_set_russian,
        'arb': arabic_to_ipa_set_msa,
        'deu': latin_to_ipa_set_german,
        'fra': latin_to_ipa_set_french,
        'tur': latin_to_ipa_set_turkish,
        'ita': latin_to_ipa_set_italian,
    }

    ipa_to_script_chars = defaultdict(lambda: defaultdict(lambda: set()))
    for lang, ipa_set in script_to_ipa_chars.items():
        for script_char, ipas in ipa_set.items():
            for ipa in ipas:
                ipa_to_script_chars[lang][ipa].add(script_char)

    return ipa_to_script_chars, script_to_ipa_chars

def get_character_set(lang):
    '''Get character set for script
    Args:
        lang: str, language code
    Returns:
        script, str, script
        char_set, character set of that script
    '''
    # Get character set for script using Unicode ranges
    lang = normalize_lang_codes(lang)
    lang_to_script = {
        "eng": "latin",
        "deu": "latin",
        "hin": "devanagari",
        "arb": "arabic",
        "rus": "cyrillic",
        "spa": "latin",
        "ind": "latin",
        "fra": "latin",
        "tur": "latin",
        "ita": "latin",
        "hat": "latin",
    }
    
    script = lang_to_script[lang]

    char_set = {
        # Include all accents for latin
        "latin": set(chr(i) for i in range(65, 91)) | set(chr(i) for i in range(97, 123)) | \
                    set(chr(i) for i in range(192, 256)),

        "devanagari": set(chr(i) for i in range(2304, 2432)),
        "arabic": set(chr(i) for i in range(1536, 1792)),
        "cyrillic": set(chr(i) for i in range(1024, 1280)),
    }

    # Add in any additional characters
    script_to_ipa_chars = ipa_char_maps()[1][lang]
    script_chars = set(script_to_ipa_chars.keys()) | char_set[script]

    return script, script_chars



def get_equivalence_classes_ipa():
    '''
    Given a set of IPA characters, group them into equivalence classes
    '''
    ipa_chars = {
        'ʝ', 'a', 'aɪ', 'ɨ', 'aː', 'bʱ', 'ju', 'ə', 'r̩', 'ɾ', 'ɕtɕ', \
            'ː', 'v', 't͡ʃ', 'ks', 't͡s', 'd͡ʒʱ', 'j', 'ʔ', 'əŋ', 'o', \
            'ts', 's', 'ʌ', 'ʏ', 'ʁ', 'ʂ', 'ð', 'je', 'in', 'iː', \
            'uː', 'kʰ', 'd', 'u', 'd̪', 'ʐ', 'ɖ', 'ɳ', 'd͡ʒɲ', 'w', \
            'ʈ', 'tˤ', 'l', 'ja', 'd̪ʱ', 'dˤ', 'q', 'ɣ', 'f', 'kʂ', \
            'oː', 't̪', 'pʰ', 'ɾ', 'χ', 'rɪ', 'yː', 'p', 'aʊ', 'ɖʱ', \
            'ɑ', 'h', 'ɡʱ', 'ʃ', 'tɕ', 'ø', 'ɭ', 't̪ʰ', 'd͡ʒ', 'ʕ', \
            'ʊ', 'ɲ', 'b', 'ɔ', 'm', 'e', 't͡ʃʰ', 't', 'i', 'ʈʰ', 'k', \
            'ɛ', 'jo', 'æ', 'r', 'x', 'əh', 'ɡ', 'z', 'un', 'ʉ', 'ʦ', 'øː', \
            'ħ', 'eː', 'ɪ', 'β', 'ŋ', 'ðˤ', 'y', 'sˤ', 'ɡz', 'θ', 'dʒ', 'n', 'kv', 'ɥ', 'ɯ', 'ø'
    }

    # Still missing from below:
    # {'jo','ju','je', 'ja', 'kv'}

    # ipa_equivalence_classes = {
    #     # Consonants:
    #     'bilabial_labiodental': {'b', 'bʱ', 'p', 'pʰ', 'ɓ', 'f', 'v', 'β', 'w'},
    #     # 'labiodental' : {'f', 'v', 'β'},
    #     'dental_alveolar': {'d', 't', 'd̪', 't̪', 's', 'z', 'ɾ', 'ɖ', 'ɭ', \
    #                      'ʈ', 'dʒ', 'dˤ', 'ɖʱ', 'ðˤ', 't̪ʰ', 'tˤ', 't͡ʃʰ', 'tɕ', 't͡s', 'ʈʰ', 'd̪ʱ', 'θ', 'ð', 'd͡ʒʱ', 'd͡ʒɲ'},
    #     'nasals': {'m', 'n', 'ɲ', 'ŋ', 'ɳ', 'd͡ʒɲ'},
    #     'postalveolar_palatal': {'ʃ', 'ʒ', 'd͡ʒ', 't͡ʃ', 't͡ɕ', 'd͡ʑ', 'ɕ', 'ʑ', 'j', 'ç', 'ɕtɕ', 'ʂ', 'ts', 'ks', 'ɡz', 'ʐ', 'kʂ', 'ʝ', 'sˤ', 'ʦ'},
    #     'retroflex': {'ɽ', 'ɭ', 'r', 'l', 'rɪ', 'r̩'},
    #     # 'palatal': {'j', 'ɲ', 'ç'},
    #     'velar': {'k', 'kʰ', 'ɡ', 'ɡʱ', 'x', 'ɣ', 'ks', 'ɡz'},
    #     'uvular_pharyngeal_glottal': {'q', 'χ', 'ʁ', 'ʕ', 'ħ', 'ʔ', 'h'},
    #     # 'pharyngeal/Glottal': {'ʕ', 'ħ', 'ʔ'},
    #     # Vowels:
    #     'front': {'i', 'e', 'ɛ', 'æ', 'ɪ', 'ø', 'y', 'œ', 'eː', 'ɛː', 'øː', 'œː', 'ʏ', 'iː', 'ʊ', 'aɪ', 'yː', 'aʊ', 'in'},
    #     'central': {'ə', 'ʉ', 'ɨ', 'ɵ', 'ɵː', 'əh', 'əŋ'},
    #     'back': {'ɑ', 'ɒ', 'ʌ', 'ɤ', 'o', 'u', 'ɔ', 'ɯ', 'uː', 'oː', 'ɔː', 'ɯː', 'uː', 'oː', 'ɔː', 'un', 'aː', 'aʊ', 'a'}
    # }

    # Equivalence classing of IPA characters
    '''
        # Bilabials
        # 'bilabial_labiodental': {'b', 'bʱ', 'p', 'pʰ', 'ɓ', 'f', 'v', 'β', 'w'},
        ## Aspiration
        {'p', 'pʰ'},
        {'b', 'bʱ'},
        {'β', 'bʱ'},
        {'ɓ', 'bʱ'}, #'ɓ' is a voiced bilabial implosive 

        ## Voicing 
        {'b', 'p'},
        {'bʱ', 'pʰ'},
        {'ɓ', 'p'}, 
        {'β', 'p'}

        ## Manner
        {'β','b', 'ɓ'},

        # Labiodentals
        ## Place (and manner)
        {'b', 'β', 'ɓ', 'v', 'w'},
        {'p', 'f', 'w'},
        
        # Dentals, alveolars, postalveolars, palatals, retroflexes
        # {'d', 't', 'd̪', 't̪', 's', 'z', 'ɾ', 'ɖ', 'ɭ', \
        # 'ʈ', 'dʒ', 'dˤ', 'ɖʱ', 'ðˤ', 't̪ʰ', 'tˤ', 't͡ʃʰ', 'tɕ', 't͡s', 'ʈʰ', 'd̪ʱ', 'θ', 'ð', 'd͡ʒʱ'},
        # {'ʃ', 'ʒ', 'd͡ʒ', 't͡ʃ', 't͡ɕ', 'd͡ʑ', 'ɕ', 'ʑ', 'j', \
        #  'ç', 'ɕtɕ', 'ʂ', 'ts', 'ks', 'ɡz', 'ʐ', 'kʂ', 'ʝ', 'sˤ', 'ʦ'},

        ## Aspiration (and place)
        {'t', 'ʈ', 't̪', 't̪ʰ', 'ʈʰ', 'θ', 'tˤ', },
        {'d', 'ɖʱ', 'd̪', 'd̪ʱ', 'ð', 'dˤ', 'ðˤ'},
        {'s', 't͡ʃʰ'},
        {'dʒ', 'd͡ʒʱ'},

        ## Voicing (and place)
        {'tˤ', 'dˤ'},
        {'d', 'd̪', 'ð', 'ɖ', 't', 't̪', 'ʈ', 'θ', 'ɾ'},
        {'s', 'z', 'ʒ', 'ʑ', 'ʐ'},
        {'t͡ɕ', 'd͡ʑ'},
        {'ks', 'ɡz'},
        
        ## Manner (and place)
        {'s', 'dʒ', 'ts', 't͡s', 'ʃ', 'ʂ', 'sˤ', 'ʦ'},
        {'ʃ', 'ʒ', 'd͡ʒ', 't͡ʃ', 't͡ɕ', 'd͡ʑ', 'ɕ', 'ʑ', 'j', 'ʐ', 'ɕtɕ', 'ʝ'},
        {'ks', 'ɡz', 'kʂ'}

        # Approximants - putting all approximants together, some place change as well
        {'ɽ', 'r', 'rɪ', 'r̩', 'ʁ', 'ɾ'},
        {'ɭ', 'l'}, 
        {'l', 'j'}, # just here because l will have nowhere to go in many languages

        # Velars
        # {'k', 'kʰ', 'ɡ', 'ɡʱ', 'x', 'ɣ', 'ks', 'ɡz'},

        ## Aspiration
        {'k', 'kʰ'},
        {'ɡ', 'ɡʱ'},

        ## Voicing
        {'ɡ', 'k'},
        {'ɡʱ', 'kʰ'},
        {'ɣ', 'x'},
        {'ɡz', 'ks'},

        ## Manner
        {'ɡ', 'ɣ', 'ɡʱ'},
        {'k', 'x', 'kʰ'},

        # Uvulars, pharyngeals, glottals
        {'q', 'χ', 'ʁ', 'ʕ', 'ħ', 'ʔ', 'h', 'əh'}, \
        
        # Nasals
        {'m', 'n', 'ɲ', 'ŋ', 'ɳ'},
        {'m', 'n', 'ɲ', 'ŋ', 'ɳ'}, \
        {'əŋ', 'in', 'un'}, \

        # Vowels
        # 'front': {'i', 'e', 'ɛ', 'æ', 'ɪ', 'ø', 'y', 'œ', 'eː', 'ɛː', 'øː', 'œː', 'ʏ', 'iː', 'ʊ', 'aɪ', 'yː', 'aʊ', 'in'},
        # 'central': {'ə', 'ʉ', 'ɨ', 'ɵ', 'ɵː', 'əh', 'əŋ'},
        # 'back': {'ɑ', 'ɒ', 'ʌ', 'ɤ', 'o', 'u', 'ɔ', 'ɯ', 'uː', 'oː', 'ø' 'ɔː', 'ɯː', 'uː', 'oː', 'ɔː', 'un', 'aː', 'aʊ', 'a'}
        {'i', 'e', 'ɛ', 'æ', 'ɪ', 'ø', 'y', 'œ', 'eː', \
         'ɛː', 'øː', 'œː', 'ʏ', 'iː', 'ʊ', 'aɪ', 'yː', 'aʊ',\
        'ə', 'ʉ', 'ɨ', 'ɵ', 'ɵː', 'əh', \
        'ɑ', 'ɒ', 'ʌ', 'ɤ', 'o', 'u', 'ɔ', 'ɯ', \
        'uː', 'oː', 'ɔː', 'ɯː', 'uː', 'oː', 'ɔː', 'aː', 'aʊ', 'a', 'ɥ'}
    '''
    ipa_equivalence_classes = \
        [ {'p', 'pʰ'}, \
        {'b', 'bʱ'}, \
        {'β', 'bʱ'}, \
        {'ɓ', 'bʱ'},  \
        {'b', 'p'}, \
        {'bʱ', 'pʰ'}, \
        {'ɓ', 'p'},  \
        {'β', 'p'}, \
        {'β','b', 'ɓ'}, \
        {'b', 'β', 'ɓ', 'v', 'w'}, \
        {'p', 'f', 'w'}, \
        {'t', 'ʈ', 't̪', 't̪ʰ', 'ʈʰ', 'θ', 'tˤ'}, \
        {'d', 'ɖʱ', 'd̪', 'd̪ʱ', 'ð', 'dˤ', 'ðˤ'}, \
        {'s', 'ʃ'}, \
        {'t͡ʃ', 't͡ʃʰ'}, \
        {'dʒ', 'd͡ʒ', 'd͡ʒʱ'}, \
        {'tˤ', 'dˤ'}, \
        {'d', 'd̪', 'ð', 'ɖ', 't', 't̪', 'ʈ', 'θ'}, \
        {'s', 'z', 'ʒ', 'ʑ', 'ʐ'}, \
        {'t͡ɕ', 'd͡ʑ'}, \
        {'ks', 'ɡz'}, \
        {'s', 'dʒ', 'ts', 't͡s', 'ʃ', 'ʂ', 'sˤ', 'ʦ'}, \
        {'ʃ', 'ʒ', 'd͡ʒ', 't͡ʃ', 't͡ɕ', 'd͡ʑ', 'ɕ', 'ʑ', 'j', 'ʐ', 'ɕtɕ', 'ʝ'}, \
        {'ks', 'ɡz', 'kʂ'}, \
        {'ɽ', 'r', 'rɪ', 'r̩', 'ʁ', 'ɾ'}, \
        {'ɭ', 'l'},  \
        {'l', 'j'}, \
        {'k', 'kʰ'}, \
        {'ɡ', 'ɡʱ'}, \
        {'ɡ', 'k'}, \
        {'ɡʱ', 'kʰ'}, \
        {'ɣ', 'x'}, \
        {'ɡz', 'ks'}, \
        {'ɡ', 'ɣ', 'ɡʱ'}, \
        {'k', 'x', 'kʰ'}, \
        {'q', 'χ', 'ʁ', 'ʕ', 'ħ', 'ʔ', 'h', 'əh'}, \
        {'m', 'n', 'ɲ', 'ŋ', 'ɳ'}, \
        {'əŋ', 'in', 'un'}, \
        {'d͡ʒɲ', 'n', 'ɲ', 'ŋ', 'ɳ'}, \
        {'i', 'e', 'ɛ', 'æ', 'ɪ', 'ø', 'y', 'œ', 'eː', \
         'ɛː', 'øː', 'œː', 'ʏ', 'iː', 'ʊ', 'aɪ', 'yː', 'aʊ',\
        'ə', 'ʉ', 'ɨ', 'ɵ', 'ɵː', 'əh', \
        'ɑ', 'ɒ', 'ʌ', 'ɤ', 'o', 'u', 'ɔ', 'ɯ', 'ø' \
        'uː', 'oː', 'ɔː', 'ɯː', 'uː', 'oː', 'ɔː', 'aː', 'aʊ', 'a', 'ɥ'}
    ]


    # Original, cleaner version by place of articulation
    # ipa_equivalence_classes = {
    #     # Consonants:
    #     'bilabial': {'b', 'bʱ', 'p', 'pʰ', 'm', 'ɓ'},
    #     'labiodental': {'f', 'v', 'β'},
    #     'dental_alveolar': {'d', 't', 'd̪', 't̪', 'd͡ʒ', 't͡ʃ', 's', 'z', 'n', 'l', 'ɾ', 'ɖ', 'ʈ', 'ɳ', 'ɭ', 'ɽ'},
    #     'nasals': {'m', 'n', 'ɲ', 'ŋ', 'ɳ'},
    #     'postalveolar': {'ʃ', 'ʒ', 't͡ɕ', 'd͡ʑ', 'ɕ', 'ʑ'},
    #     'retroflex': {'ɽ', 'ʈ', 'ɳ', 'ɭ'},
    #     'palatal': {'j', 'ɲ', 'ç'},
    #     'velar': {'k', 'kʰ', 'ɡ', 'ɡʱ', 'ŋ', 'x', 'ɣ'},
    #     'uvular': {'q', 'χ', 'ʁ'},
    #     'pharyngeal/Glottal': {'ʕ', 'ħ', 'ʔ'},
    #     # Vowels:
    #     'front': {'i', 'e', 'ɛ', 'æ', 'ɪ', 'ø', 'y', 'œ', 'eː', 'ɛː', 'øː', 'œː'},
    #     'central': {'ə', 'ʉ', 'ɨ', 'ɵ', 'ɵː'},
    #     'back': {'ɑ', 'ɒ', 'ʌ', 'ɤ', 'o', 'u', 'ɔ', 'ɯ', 'uː', 'oː', 'ɔː', 'ɯː', 'uː', 'oː', 'ɔː'}
    # }

    # Equivalence class per character
    ## Every character can go to another other character that it appears in an eqv set with
    ipa_equivalence_classes_per_char = defaultdict(lambda: set())
    for chars in ipa_equivalence_classes:
        for char in chars:
            ipa_equivalence_classes_per_char[char].update(chars)

    # print(f"Number of IPA characters: {len(ipa_chars)}")
    # print("Vowels:", vowels)
    # print("Consonants:", consonants)
    # print(f"Number of vowels: {len(vowels)}")
    # print(f"Number of consonants: {len(consonants)}")

    return ipa_chars, ipa_equivalence_classes, ipa_equivalence_classes_per_char

# all_ipa_chars = set()
# for lang in {"eng", "deu", "hin", "ara", "rus", "spa", "ind"}:
#     script, char_set = get_character_set(lang)
#     script_to_ipas = ipa_char_maps(lang)
#     for char, ipas in script_to_ipas.items():
#         all_ipa_chars.update(ipas)

# print(all_ipa_chars)
# print(get_equivalence_classes_ipa())
