def flores_code_to_langname(code):
    iso3_to_lang = {
        "hne_Deva": "Chhattisgarhi",
        "bho_Deva": "Bhojpuri",
        "mag_Deva": "Magahi",
        "mai_Deva": "Maithili",
        "hin_Deva": "Hindi",
        "tur_Latn": "Turkish",
        "uzn_Latn": "Uzbek",
        "tuk_Latn": "Turkmen",
        "azj_Latn": "Azerbaijani",
        "crh_Latn": "Crimean Tatar",
        "spa_Latn": "Spanish",
        "fra_Latn": "French",
        "por_Latn": "Portuguese",
        "ita_Latn": "Italian",
        "ron_Latn": "Romanian",
        "glg_Latn": "Galician",
        "cat_Latn": "Catalan",
        "oci_Latn": "Occitan",
        "ast_Latn": "Asturian",
        "lmo_Latn": "Lombard",
        "vec_Latn": "Venetian",
        "scn_Latn": "Sicilian",
        "srd_Latn": "Sardinian",
        "fur_Latn": "Friulian",
        "lij_Latn": "Ligurian",
        "ind_Latn": "Indonesian",
        "jav_Latn": "Javanese",
        "sun_Latn": "Sundanese",
        "smo_Latn": "Samoan",
        "mri_Latn": "Maori",
        "ceb_Latn": "Cebuano",
        "zsm_Latn": "Malay",
        "tgl_Latn": "Tagalog",
        "ilo_Latn": "Ilokano",
        "fij_Latn": "Fijian",
        "plt_Latn": "Plateau Malagasy",
        "pag_Latn": "Pangasinan",
        "arb_Arab": "Arabic",
        "acm_Arab": "Iraqi Arabic",
        "acq_Arab": "Ta'izzi-Adeni Arabic",
        "aeb_Arab": "Tunisian Arabic",
        "ajp_Arab": "South Levantine Arabic",
        "apc_Arab": "North Levantine Arabic",
        "ars_Arab": "Najdi Arabic",
        "ary_Arab": "Moroccan Arabic",
        "arz_Arab": "Egyptian Arabic"
    }
    return iso3_to_lang[code]

def flores_code_to_hrln(code):
    hrln2crls = {
        "hin_Deva": ["hne_Deva", "bho_Deva", "mag_Deva", "mai_Deva", "hin_Deva"],
        "tur_Latn": ["tur_Latn", "uzn_Latn", "tuk_Latn", "azj_Latn", "crh_Latn"],
        "ita_Latn": ["spa_Latn", "fra_Latn", "por_Latn", "ita_Latn", "ron_Latn", "glg_Latn", "cat_Latn", "oci_Latn", "ast_Latn", "lmo_Latn", "vec_Latn", "scn_Latn", "srd_Latn", "fur_Latn", "lij_Latn"],
        "ind_Latn": ["ind_Latn", "jav_Latn", "sun_Latn", "smo_Latn", "mri_Latn", "ceb_Latn", "zsm_Latn", "tgl_Latn", "ilo_Latn", "fij_Latn", "plt_Latn", "pag_Latn"],
        "arb_Arab": ["arb_Arab", "acm_Arab", "acq_Arab", "aeb_Arab", "ajp_Arab", "apc_Arab", "ars_Arab", "ary_Arab", "arz_Arab", \
                     "cai", "dam", "doh", "fes", "jer", "kha", "msa", "riy", "san", "tri", "tun"] # FloRes codes, MADAR codes
        }

    for hrln, crls in hrln2crls.items():
        if code in crls:
            return hrln, flores_code_to_langname(hrln)

def get_crls(hrln):
    hrln2crls = {
        "hin": ["hne_Deva", "bho_Deva", "mag_Deva", "mai_Deva", "hin_Deva"],
        "tur": ["tur_Latn", "uzn_Latn", "tuk_Latn", "azj_Latn", "crh_Latn"],
        "ita": ["spa_Latn", "fra_Latn", "por_Latn", "ita_Latn", "ron_Latn", "glg_Latn", "cat_Latn", "oci_Latn", "ast_Latn", "lmo_Latn", "vec_Latn", "scn_Latn", "srd_Latn", "fur_Latn", "lij_Latn"],
        "ind": ["ind_Latn", "jav_Latn", "sun_Latn", "smo_Latn", "mri_Latn", "ceb_Latn", "zsm_Latn", "tgl_Latn", "ilo_Latn", "fij_Latn", "plt_Latn", "pag_Latn"],
        "arb": ["arb_Arab", "acm_Arab", "acq_Arab", "aeb_Arab", "ajp_Arab", "apc_Arab", "ars_Arab", "ary_Arab", "arz_Arab"],
        "arb_madar": ["cai", "dam", "doh", "fes", "jer", "kha", "msa", "riy", "san", "tri", "tun"],
        "hat": ["gcf", "mart1259", "acf", "gcr", "lou", "mfe", "rcf", "crs", "hat"]
        }
    return hrln2crls[hrln]