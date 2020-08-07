#!/bin/bash
  
DATA_DIR=$BIO_ELECTRA_HOME/electra/data
source $BIO_ELECTRA_HOME/venv/bin/activate

python run_finetuning.py --data-dir $DATA_DIR --model-name electra_small --hparams '{"model_size": "small", "task_names": ["gad"], "max_seq_length":128, "train_batch_size":32, "do_train":true, "use_tfrecords_if_existing": true, "num_trials": 10, "write_test_outputs": true, "n_writes_test": 10}'

