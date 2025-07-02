def get_default_parameters(lang, text_file):

    params = {
        "lexical_aug": {
            "lang": lang,
            "theta_content_global": 0.001,
            "theta_func_global": 0.8,
            "text_file": text_file
        },
        "morph_aug": {
            "lang": lang,
            "theta_morph_global": 0.5,
            "text_file": text_file
        },
        "phonological_aug": {
            "lang": lang,
            "theta_phon": 0.07,
            "text_file": text_file
        },
        "random_char_aug": {
            "lang": lang,
            "text_file": text_file
        },
        "random_word_aug": {
            "lang": lang,
            "theta_random_word": 0,
            "text_file": text_file
        },
        # "semantic_aug": {
        #     "lang": "hi",
        #     "theta_semantic_global": 0,
        #     "text_file": "/export/b08/nbafna1/data/wikimatrix/en-hi/WikiMatrix.en-hi.hi"
        # }
    }

    return params