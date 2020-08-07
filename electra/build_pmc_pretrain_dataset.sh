#!/bin/bash

DATA_DIR=$BIO_ELECTRA_HOME/electra/data
source $BIO_ELECTRA_HOME/venv/bin/activate
WD=$BIO_ELECTRA_HOME/electra/data/electra_pretraining

python build_pretraining_dataset.py --corpus-dir $WD/pmc_abstracts --vocab-file $DATA_DIR/pmc_2017_abstracts_wp_vocab_sorted.txt --output-dir $WD/electra_train_data --max-seq-length 256 --no-lower-case


