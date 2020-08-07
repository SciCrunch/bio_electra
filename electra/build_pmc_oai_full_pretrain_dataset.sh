#!/bin/bash

DATA_DIR=$BIO_ELECTRA_HOME/electra/data
source $BIO_ELECTRA_HOME/venv/bin/activate
WD=$BIO_ELECTRA_HOME/data/electra_pmc_oai_pretraining

python build_pretraining_dataset.py --corpus-dir  $WD --vocab-file $DATA_DIR/pmc_2017_abstracts_wp_vocab_sorted.txt --output-dir $DATA_DIR/oai_electra_train_data --max-seq-length 256 --no-lower-case


