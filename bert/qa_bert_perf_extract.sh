#!/bin/bash

export ROOT_DIR=/tmp/qa_bert_batch

for i in 0 1 2 3 4 5 6 7 8 9; do
    python $BIO_ELECTRA_HOME/bert/squad/evaluate-v1.1.py  $BIO_ELECTRA_HOME/electra/data/finetuning_data/bioasdev.json $ROOT_DIR/run_$i/predictions.json
done
