#!/bin/bash

source $BIO_ELECTRA_HOME/venv/bin/activate

export BERT_BASE_DIR=$BIO_ELECTRA_HOME/models/cased_L-12_H-768_A-12
export DATA_DIR=$BIO_ELECTRA_HOME/electra/data/finetuning_data/bioasq
export OUT_DIR_ROOT=/tmp/qa_bert_batch

[ -d $OUT_DIR_ROOT ] || mkdir $OUT_DIR_ROOT

seeds=(936084 22968 154859 808249 172158 828438 94143 389173 962489 797140)

for i in ${!seeds[@]}; do
  OUT_DIR=$OUT_DIR_ROOT/run_$i
  echo "OUTDIR=$OUT_DIR"
  SEED=${seeds[$i]}
  echo "SEED=$SEED"
  python run_squad.py \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --do_train=True \
      --train_file=$DATA_DIR/train.json \
      --do_predict=True \
      --predict_file=$DATA_DIR/dev.json \
      --train_batch_size=12 \
      --learning_rate=3e-5 \
      --num_train_epochs=2.0 \
      --max_seq_length=192 \
      --doc_stride=128 \
      --seed $SEED \
      --output_dir=$OUT_DIR

  python bioasq_qa_result_gen.py -i $DATA_DIR/dev.json -r $OUT_DIR/nbest_predictions.json -o $OUT_DIR/qa_nbest.json

done

