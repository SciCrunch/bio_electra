#!/bin/bash

source $BIO_ELECTRA_HOME/venv/bin/activate

export BERT_BASE_DIR=$BIO_ELECTRA_HOME/models/cased_L-12_H-768_A-12
export DATA_DIR=$BIO_ELECTRA_HOME/electra/data/finetuning_data/yesno
export OUT_DIR_ROOT=/tmp/yes_no_batch

[-d $OUT_DIR_ROOT] || mkdir $OUT_DIR_ROOT

seeds=(936084 22968 154859 808249 172158 828438 94143 389173 962489 797140)

for i in ${!seeds[@]}; do
  OUT_DIR=$OUT_DIR_ROOT/run_$i
  echo "OUTDIR=$OUT_DIR"
  SEED=${seeds[$i]}
  echo "SEED=$SEED"
  python run_bioasq_classifier.py \
    --task_name=bioasq \
    --do_predict=true \
    --data_dir=$DATA_DIR \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$OUT_DIR \
    --max_seq_length=192 \
    --train_batch_size=12 \
    --learning_rate=1e-4 \
    --num_train_epochs=3.0 \
    --output_dir=$OUT_DIR

  python bioasq_yesno_f1_perf.py -i $DATA_DIR/test.tsv -r $OUT_DIR/test_results.tsv -o /tmp/bert_yesno_f1_results.csv
done

