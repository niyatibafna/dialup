## Introduction

D-->M is an inference time technique to replace low-resource language (LRL) words in the input with their high-resource language (HRL) equivalents, using bilingual dictionaries.

## Languages supported
The following languages in each language family are supported
**Austronesian Language Family**
* Indonesian (`ind_Latn`)
* Javanese (`jav_Latn`)
* Sundanese (`sun_Latn`)
* Samoan (`smo_Latn`)
* Maori (`mri_Latn`)
* Cebuano (`ceb_Latn`)
* Standard Malay (`zsm_Latn`)
* Tagalog (`tgl_Latn`)
* Ilocano (`ilo_Latn`)
* Fijian (`fij_Latn`)
* Plateau Malagasy (`plt_Latn`)
* Pangasinan (`pag_Latn`)

**Arabic Language Family**
* Modern Standard Arabic (`arb_Arab`)
* Mesopotamian Arabic (`acm_Arab`)
* Ta'izzi-Adeni Arabic (`acq_Arab`)
* Tunisian Arabic (`aeb_Arab`)
* South Levantine Arabic (`ajp_Arab`)
* North Levantine Arabic (`apc_Arab`)
* Najdi Arabic (`ars_Arab`)
* Moroccan Arabic (`ary_Arab`)
* Egyptian Arabic (`arz_Arab`)

**Romance Language Family**
* Italian (`ita_Latn`)
* Spanish (`spa_Latn`)
* French (`fra_Latn`)
* Portugese (`por_Latn`)
* Romanian (`ron_Latn`)
* Galician (`glg_Latn`)
* Catalan (`cat_Latn`)
* Occitan (`oci_Latn`)
* Asturian (`ast_Latn`)
* Lombard (`lmo_Latn`)
* Venetian (`vec_Latn`)
* Sicilian (`scn_Latn`)
* Sardinian (`srd_Latn`)
* Friulian (`fur_Latn`)
* Ligurian (`lij_Latn`)

**Turkic Language Family**
* Turkish (`tur_Latn`)
* Northern Uzbek (`uzn_Latn`)
* Turkmen (`tuk_Latn`)
* North Azerbaijani (`azj_Latn`)
* Crimean Tatar (`crh_Latn`)

**Indic Language Family**
* Hindi (`hin_Deva`)
* Bhojpuri (`bho_Deva`)
* Magahi (`mag_Deva`)
* Maithili (`mai_Deva`)

**Creole Language Family**
* Haitian (`hat`)
* Saint Lucian Patois (`acf`)
* Mauritian (`mfe`)
* Seychellosis (`crs`)

## Bilingual lexicons 
These lexicons translate a word from an LRL to their semantically-equivalent counterparts in a closely-related HRL. The compendium of lexicons in this repository come from a variety of resources, such as [Swadesh lists](https://en.wiktionary.org/wiki/Category:Swadesh_lists_by_language), [PanLex](https://panlex.org/), and [Art Dieli](http://www.dieli.net/SicilyPage/SicilianLanguage/Vocabulary.html). It should be noted that we did not include translations from *all* sources. Rather, we prioritized more reliable lexicons over others. For example, if the Italian translation of a Sicilian word appeared in Art Dieli's lexicons, PanLex, and a statistically-aligned lexicon, we selected the translations from Art Dieli's lexicons. Art Dieli's lexicons was hand curated by an individual with knowledge of Sicilian culture and language. Similarly, if a Hindi translation of a Bhojpuri word appeared in Swadesh list, PanLex, a BLI-induced lexicon, and a statistically-aligned lexicon, the translations from the Swadesh list were selected over ones from other sources. PanLex translations were preferred over BLI-induced and statistically-aligned lexicons. 

You may notice that each high-resource language word is associated with a number, a value that can be thought of as a "confidence score." Because our lexicons were aggregated over many sources which used various confidence scoring approaches, you will see a wide range of such values (from 1 to somewhere in the 100s).

## Run on your input
The SLURM script that runs the DTM evaluation code is `dtm/evaluate_dtm.sh`. The variable `HRL_LRL_PAIRS` stores the high-resource language and associated low-resource language pairs we seek to evaluate. This list can be expanded to include additional pairs. THe SLURM option `--array` controls which language(s) to evaluate.

You may customize this script by modifying the flags described below.
* `exp_key`: the name of the experiment being run. This flag concatenates the name of the current model being evaluated and the denoising schema being utilized.
* `hrl`: a high-resource language. This flag can be set to Hindi (`hin`), Turkish (`tur`), Italian (`ita`), Indonesian (`ind`), Modern Standard Arabic (`arb`), and Haitan Creole (`hat`)
* `crl`: a langugae that is closely-rekated to the `hrl`. For example, this flag can be set to Chattisgarhi (`hne_Deva`), Northern Uzbek (`uzn_Latn`), Galician (`glg_Latn`), etc.
* `model_name`: the model you would like to run evaluation on. This is currently set to either `aya-23-8b` or `m2mL`
* `bilingual_lexicon_path`: base folder path for bilingual lexicons. By default, this is set to `dtm/lexicons/`.
* `flores_dir`: folder path to FloRes-200 dataset.
* `kreyolmt_dir`: folder path to Creole evaluation dataset.
* `madar_dir`: folder path to the Arabic MADAR dataset. While this is currently not set, it can be added to evaluate on this dataset. The Arabic subset of FloRes 200 will be ignored.
* `denoise_func`: controls how the DTM input is constructed, specifically which words get "swapped out". For example, if this flag is set to `functional`, only LRL functional words are swapped for their HRL equivalents.
* `batch_size `: controls how many DTM inputs are processed at one time.
* `mt_outputs_dir`: folder path where the English translation for the DTM input are stored. Best used for **qualitative** analysis.
* `results_dir `: folder path where the BLEU scores for each language are stored. Best used for **quantitative** analysis.

## Add a new bilingual lexicon / language
Lexicons are sorted by language pair and separated into `functional`, `content`, and `all` (a combination of `functional` and `content`). Should you wish to evaluate an LRL-HRL pair that doesn't yet exist, you will have to create an additional folder that follows the `lrl-hrl` naming convention prior to adding lexicons. In addition to the lexicon that contains all known LRL-HRL translation pairs, you will have to create two more lexicons: one containing content words and another containing functional words.

For the lexicon to be processed correctly, it must be stored as a JSON and follow the format below:
```
{
    <word in LRL>: {
        <translated word in HRL>: <confidence score>,
        <translated word in HRL>: <confidence score>,
        <translated word in HRL>: <confidence score>,
        ...
    }, ...
}
```

### Leveraging UD treebanks
Let's say you acquire a Universal Dependency treebank for an HRL that we have not covered. Or, you would like to leverage a treebank we already use to create lexicons for your own LRL. You can accomplish this by:
1. Add the treebank under `dtm/ud_closed_class_wordlists` (if it doesn't already exist)
2. Add the associated HRL and path to the treebank to `ud_wordlist_paths` in `dtm/utils/paths.py` (if it doesn't exist already)
3. Iterate through the lexicon contianing all translations, checking if the associated HRL word is functional using `Denoiser.is_hrl_word_functional()`. Should this function return `true`, bin the LRL and associated HRL word in the functional lexicon. Otherwise, bin the LRL and associated HRL word in the content lexicon.