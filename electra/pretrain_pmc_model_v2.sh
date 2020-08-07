#!/bin/bash

DATA_DIR=$BIO_ELECTRA_HOME/electra/data
source $BIO_ELECTRA_HOME/venv/bin/activate

python run_pretraining.py --data-dir $DATA_DIR --model-name pmc_electra_small_v2 --hparams pmc_config_v2.json 



