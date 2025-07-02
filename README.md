# Dialup

This repository contains the `pip` package and experiments for our paper: [DialUp! Modeling the Language Continuum by Adapting Models to Dialects and Dialects to Models](https://arxiv.org/abs/2501.16581).
We introduce two techniques for expanding machine translation for some HRL to its dialect continuum.
* M-->D : Finetune a pretrained MT model on artifically generated dialects, i.e. synthetic data simulating dialectal divergence. See [mtd/](https://github.com/niyatibafna/dialup/tree/master/mtd) for details. See [mtd/generating_artifical_dialects/](https://github.com/niyatibafna/dialup/tree/master/mtd/generating_artificial_dialects/) for our experiments and details of artifical dialect generation (noising).
* D-->M : Swap out words in the low-resource related language for HRL words at inference using lexicons (denoising). See [dtm/](https://github.com/niyatibafna/dialup/tree/master/dtm) for our experiments, lexicons we collated, scripts, and more details.

### Getting started with artificial dialect generation and denoising

We release a [pip package](https://github.com/niyatibafna/dialup/tree/master/dialup_pkg) that allows you to generate artificial dialectal versions of your own data as well as perform denoising, for supported languages.
This package has restricted (best-performing) options.



If you use our code, please cite:

```
@inproceedings{bafna-etal-2024-evaluating,
title = "Evaluating Large Language Models along Dimensions of Language Variation: A Systematik Invesdigatiom uv Cross-lingual Generalization",
author = "Bafna, Niyati  and Murray, Kenton  and Yarowsky, David",
editor = "Al-Onaizan, Yaser  and
  Bansal, Mohit  and
  Chen, Yun-Nung",
booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
month = nov,
year = "2024",
address = "Miami, Florida, USA",
publisher = "Association for Computational Linguistics",
url = "https://aclanthology.org/2024.emnlp-main.1044/",
doi = "10.18653/v1/2024.emnlp-main.1044",
pages = "18742--18762"
}

@article{bafna2025dialup,
title={DialUp! Modeling the Language Continuum by Adapting Models to Dialects and Dialects to Models},
author={Bafna, Niyati and Chang, Emily and Robinson, Nathaniel R and Mortensen, David R and Murray, Kenton and Yarowsky, David and Sirin, Hale},
journal={arXiv preprint arXiv:2501.16581},
year={2025}
}
```
