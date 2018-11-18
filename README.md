# Description

This folder contains the code and preprocessed data files
for the paper
"Exploring Different Dimensions of Attention for Uncertainty Detection"
by Heike Adel and Hinrich Schuetze.

To run the CNN code with, e.g., external attention on the input, simply type:
python -u train_CNN.py configs/config_CNN_extAtt_onInp_wiki

To run the RNN code with, e.g., external attention on the input, type:
python -u train_RNN.py configs/config_RNN_extAtt_onInp_wiki

The config files *_wiki will train and evaluate on the Wikipedia dataset
of CoNLL 2010 Hedge Cue Detection Task [Farkas et al. 2010],
the config files *_bio will train and evaluate on the Biomedical dataset
of the same shared task.

The shared task data is publicly available. In this folder, we only
include our preprocessed versions of it (tokenized + represented by
word embeddings).

Tokenization has been done with the Stanford tokenizer [Manning et al. 2014].
Our script for the other preprocessing steps is createDataStream_uncertainty_blocks.py
It can be run on other data as follows:
python -u createDataStream_uncertainty_blocks.py config_newData


# Contact

If you have questions, please contact heike.adel@ims.uni-stuttgart.de


# Citation

If you use code from this folder for your work, please cite

Heike Adel and Hinrich Schuetze, "Exploring Different Dimensions of Attention for Uncertainty Detection", in EACL 2017

Bibtex:
`@inproceedings{adel2017exploring,
  authors = {Heike Adel and Hinrich Sch\"{u}tze},
  title = {Exploring Different Dimensions of Attention for Uncertainty Detection},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {22--34}
}`
