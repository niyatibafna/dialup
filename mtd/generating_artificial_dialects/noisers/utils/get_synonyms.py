def get_synonyms(word, lang, **kwargs):

    if lang == "hin":
        hin_wn = kwargs.get("hin_wn", None) # IndoWordNet object

        assert hin_wn is not None, "Please provide an instance of IndoWordNet object"
        try:
            synsets = hin_wn.synsets(word)
        except KeyError:
            return []
        all_lemmas = [l for syn in synsets for l in syn.lemma_names() if (l!=word and len(l.split(" "))==1)]
        return all_lemmas

    elif lang == "tur":
        
        tur_wn = kwargs.get("tur_wn", None) # Turkish WordNet object
        assert tur_wn is not None, "Please provide an instance of Turkish WordNet object"
        try:
            synsets = tur_wn.getSynSetsWithLiteral(word)
        except KeyError:
            return []
        synonyms = set()
        for synset in synsets:
            synonym_obj = synset.getSynonym()
            literals = [synonym_obj.getLiteral(i).name for i in range(synonym_obj.literalSize())]
            synonyms.update(literals)

        if word in synonyms:
            synonyms.remove(word) 

        return list(synonyms)

    elif lang in {"arb", "ita", "ind"}:

        wn = kwargs.get("wn", None)
        wn_key = kwargs.get("wn_key", None)
        assert wn is not None, "Please provide an instance of WordNet object"

        synonyms = set()  # Use a set to avoid duplicate synonyms
        for synset in wn.synsets(word, lexicon=wn_key):
            for lemma in synset.lemmas():
                synonyms.add(str(lemma))  # Add lemma names (synonyms) to the set
        return list(synonyms)  # Convert to list for display
