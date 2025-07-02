import os

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "ud_closed_class_wordlists")

ud_wordlist_paths ={
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
    "hat": f"{OUTDIR}/ht_nate-train.json",
}