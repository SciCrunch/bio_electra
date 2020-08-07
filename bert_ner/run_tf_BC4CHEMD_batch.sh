#!/bin/bash
source $BIO_ELECTRA_HOME/tf2_venv/bin/activate

export MAX_LENGTH=128
export BERT_MODEL=bert-base-cased
export ROOT_DIR=$BIO_ELECTRA_HOME/bert_ner
export DATA_DIR=$ROOT_DIR/data/bc4chemd
export OUTPUT_DIR_ROOT=bc4chemd_models
export BATCH_SIZE=12
export NUM_EPOCHS=3
export SAVE_STEPS=5000

[[ -f $DATA_DIR/labels.txt ]] || cat $DATA_DIR/train.txt $DATA_DIR/dev.txt $DATA_DIR/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $DATA_DIR/labels.txt

[-d $OUTPUT_DIR_ROOT] || mkdir $OUTPUT_DIR_ROOT

seeds=(936084 22968 154859 808249 172158 828438 94143 389173 962489 797140)

for i in ${!seeds[@]}; do
  OUT_DIR=$OUTPUT_DIR_ROOT/run_$i
  echo "OUTDIR=$OUT_DIR"
  SEED=${seeds[$i]}
  echo "SEED=$SEED"
  python -u  run_tf_ner.py \
    --data_dir $DATA_DIR \
    --labels $DATA_DIR/labels.txt \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUT_DIR \
    --max_seq_length  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --do_train \
    --do_eval \
    --do_predict

done
