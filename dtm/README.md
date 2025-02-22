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
These lexicons translate a word from an LRL to its semantically equivalent counterparts in a closely related HRL. The compendium of lexicons in this repository come from a variety of resources, such as [Swadesh lists](https://en.wiktionary.org/wiki/Category:Swadesh_lists_by_language), [PanLex](https://panlex.org/), and [Art Dieli](http://www.dieli.net/SicilyPage/SicilianLanguage/Vocabulary.html). Additionally, we use [Bilingual Lexicon Induction](https://aclanthology.org/2024.lrec-main.1526/) and [statistical alignment](https://aclanthology.org/N13-1073.pdf) to create these lexicons.

It should be noted that we did not include translations from *all* sources. Rather, we prioritized more reliable lexicons over others. For example, if the Italian translation of a Sicilian word appeared in Art Dieli's lexicons, PanLex, and a statistically-aligned lexicon, we selected the translations from Art Dieli's lexicons. Art Dieli's lexicons were hand-curated by an individual with knowledge of Sicilian culture and language. Similarly, if a Hindi translation of a Bhojpuri word appeared in a Swadesh list, PanLex, BLI-induced lexicon, and statistically-aligned lexicon, the translations from the Swadesh list were selected over ones from other sources. PanLex translations were preferred over BLI-induced and statistically-aligned lexicons. 

You may notice that each high-resource language word is associated with a number, a value that can be thought of as a "confidence score." Because our lexicons were aggregated over many sources that used various confidence scoring approaches, you will see a wide range of such values (from 1 to somewhere in the 100s).

## Run on your input
You can run D-->M on your  language pair and model with `evaluate_dtm.py`. Set the following options:

* `exp_key`: the name of the experiment being run. This flag concatenates the name of the current model being evaluated and the denoising schema being utilized.
* `hrl`: a high-resource language. This flag can be set to Hindi (`hin`), Turkish (`tur`), Italian (`ita`), Indonesian (`ind`), Modern Standard Arabic (`arb`), and Haitan Creole (`hat`)
* `crl`: a language that is closely related to the `hrl`. For example, this flag can be set to Chattisgarhi (`hne_Deva`), Northern Uzbek (`uzn_Latn`), Galician (`glg_Latn`), etc.
* `model_name`: the model you would like to evaluate. This is currently set to either `aya-23-8b` or `m2mL`
* `model_path`: path to a finetuned checkpoint of the above model (if relevant).
* `lora`: If `model_path` is set, this option sets whether the checkpoint was finetuned with LoRA.
* `bilingual_lexicon_path`: base folder path for bilingual lexicons. By default, this is set to `dtm/lexicons/`.
* `flores_dir`: folder path to FloRes-200 dataset.
* `kreyolmt_dir`: folder path to Creole evaluation dataset.
* `madar_dir`: folder path to the Arabic MADAR dataset. While this is currently not set, it can be added to evaluate this dataset. If set, the Arabic subset of FloRes 200 will be ignored.
* `denoise_func`: controls how the DTM input is constructed, specifically which words get "swapped out". For example, if this flag is set to `functional`, only LRL functional words are swapped for their HRL equivalents. Must be one of `functional`, `content`, `all`.
* `batch_size `: controls how many DTM inputs are processed at one time.
* `mt_outputs_dir`: folder path where the output translations are stored. 
* `results_dir `: folder path where the evaluation results for each language are stored. 

See an example run in `evaluate_dtm.sh`.

## Add a new bilingual lexicon / language
Should you wish to run D-->M on an LRL-HRL pair for which we don't have lexicons, you can create your own `lrl-hrl` lexicon in the `lexicons/` folder.

All lexicons must be stored as a JSON and follow the format below:
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

Lexicons are sorted by language pair and separated into `functional`, `content`, and `all` (a combination of `functional` and `content`). 
In our experiments, we find that depending on the language pair, it's often very useful to only switch out functional words, for example.
In case you want to use the `functional` strategy, you will further need to create the `functional` and `content` subsets of this lexicons.


### Creating `functional` and `content` subsets of the lexicon

Here are the steps for creating these sublists:
1) Curate a list of functional words, or closed class words in the HRL. Please see [here](https://github.com/niyatibafna/dialup/tree/dtm/mtd/generating_artificial_dialects) for notes on how to do this. If you can do this for your LRL of course, that is even better. If not, we will use our LRL-HRL bilingual lexicon to project this annotation on to LRL words.
2) Use the above list to separate out functional words from your collected LRL-HRL lexicon. Anything not identified as a function word is labeled a content word. (Check out `Denoisers.is_lrl_word_functional`.)




