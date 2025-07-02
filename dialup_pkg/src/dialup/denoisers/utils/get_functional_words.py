import json
from importlib.resources import files as pkg_files

OUTDIR = "data/ud_closed_class_wordlists"

ud_wordlists_paths = {
    "hin": f"{OUTDIR}/hi_hdtb-ud-train.json",
    "deu": f"{OUTDIR}/de_hdt-ud-train.json",
    "arb": f"{OUTDIR}/ar_padt-ud-train.json",
    "ind": f"{OUTDIR}/id_gsd-ud-train.json",
    "eng": f"{OUTDIR}/en_ewt-ud-train.json",
    "rus": f"{OUTDIR}/ru_syntagrus-ud-train.json",
    "spa": f"{OUTDIR}/es_gsd-ud-train.json",
    "fra": f"{OUTDIR}/fr_gsd-ud-train.json",
    "tur": f"{OUTDIR}/tr_imst-ud-train.json",
    "ita": f"{OUTDIR}/it_isdt-ud-train.json",
    "hat": f"{OUTDIR}/ht_kreolmt-train.json",
}

closed_class_tags = ['ADP', 'AUX', 'CCONJ', 'DET', 'PART', 'PRON', 'SCONJ']


def get_tag2wordlist(lang):
    '''Get tag2wordlist from the JSON file'''
    file_path = pkg_files("dialup") / ud_wordlists_paths[lang]
    with open(file_path) as f:
        tag2wordlist = json.load(f)
    return tag2wordlist
