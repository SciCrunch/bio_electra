#!/bin/bash

source $BIO_ELECTRA_HOME/venv/bin/activate

export BERT_BASE_DIR=$BIO_ELECTRA_HOME/models/cased_L-12_H-768_A-12
export DATA_DIR=$BIO_ELECTRA_HOME/electra/data/finetuning_data/gad
export OUT_DIR_ROOT=/tmp/gad_batch

[-d $OUT_DIR_ROOT] || mkdir $OUT_DIR_ROOT

seeds=(936084 22968 154859 808249 172158 828438 94143 389173 962489 797140)

for i in ${!seeds[@]}; do
  OUT_DIR=$OUT_DIR_ROOT/run_$i
  echo "OUTDIR=$OUT_DIR"
  SEED=${seeds[$i]}
  echo "SEED=$SEED"
  python run_bioasq_classifier.py \
    --task_name=biore \
    --do_predict=true \
    --data_dir=$DATA_DIR \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$OUT_DIR \
    --max_seq_length=128 \
    --train_batch_size=24 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$OUT_DIR

done

