def get_character_set(lang):
    '''Get character set for script
    Args:
        lang: str, language code
    Returns:
        script, str, script
        char_set, character set of that script
    '''
    # Get character set for script using Unicode ranges
    lang_to_script = {
        "eng": "latin",
        "deu": "latin",
        "hin": "devanagari",
        "ara": "arabic",
        "rus": "cyrillic",
        "esp": "latin",
        "en": "latin",
        "de": "latin",
        "hi": "devanagari",
        "ar": "arabic",
        "ru": "cyrillic",
        "es": "latin",
    }

    
    script = lang_to_script[lang]

    char_set = {
        "latin": set(chr(i) for i in range(65, 91)) | set(chr(i) for i in range(97, 123)),
        "devanagari": set(chr(i) for i in range(2304, 2432)),
        "arabic": set(chr(i) for i in range(1536, 1792)),
        "cyrillic": set(chr(i) for i in range(1024, 1280)),
    }
    return script, char_set[script]


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
