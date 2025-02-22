This is the code for our paper: [DialUp! Modeling the Language Continuum by Adapting Models to Dialects and Dialects to Models](https://arxiv.org/abs/2501.16581)
We introduce two techniques for expanding machine translation for some HRL to its dialect continuum.
* M-->D : Finetune a pretrained MT model on synthetic data simulating dialectal divergence.
* D-->M : Swap out words in the low-resource related language for HRL words at inference.

See `mtd/` and `dtm/` for more details and scripts for these techniques.