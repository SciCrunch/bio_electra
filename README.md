
# Bio-Electra

```bash
export BIO_ELECTRA_HOME=/full/path/to/bio_electra/repository
```

### GPU requirements
For anything besides the transformers package used for BERT NER experiments, you need Tensorflow 1.15 and CUDA 10.0 for GPU.
For BERT NER experiments on GPU, you need Tensorflow 2+ and Cuda 10.1 (i.e. another (virtual)  machine) due to tranformers Python library requirements.


### Setup virtual environment

Ensure you have virtual environment support (e.g. for Ubuntu)
```
sudo apt-get install python3-venv
```

```
python3 -m venv --system-site-packages $BIO_ELECTRA_HOME/venv
```

```
source $BIO_ELECTRA_HOME/venv/bin/activate

pip install --upgrade pip
pip install tensorflow-gpu==1.15
pip install sklearn
pip install hyperopt

```


### For BERT NER tests with Huggingface transformers python package
```
python3 -m venv --system-site-packages $BIO_ELECTRA_HOME/tf2_venv
source $BIO_ELECTRA_HOME/tf2_venv/bin/activate
pip install -U pip
pip install tensorflow-gpu==2.1
pip install transformers
pip install fastprogress
pip install seqeval
pip install torch torchvision

```
## Download BERT BASE and Electra Small++ models

* [BERT BASE  cased_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)
* [Electra Small++](https://storage.googleapis.com/electra-data/electra_small.zip)


## Pretraining

For pretraining, you need the prepare your corpus into files one line per sentence and documents separated by an empty line and put them under a single directory.
### Bio-ELECTRA
The corpus is comprised of all PubMed abstracts with PMID >= 10,000,000 (19.2 million abstracts). 
An example pretraining configuration is in the `pmc_config.json.example` file. Please copy this file to `pmc_config.json` and adjust the full paths according to your system's directory structure. The pretraining takes 3 weeks on a RTX 2070 8GB GPU.
Afterwards, assuming all of the preprocessed abstract files are under `$BIO_ELECTRA_HOME/electra/data/electra_pretraining/pmc_abstracts`, you can run the following to generate the Bio-ELECTRA language representation model.
```
cd $BIO_ELECTRA_HOME/electra
./build_pmc_pretrain_dataset.sh
./pretrain_pmc_model.sh
```
### Bio-ELECTRA++
The corpus is all open access full papers from PubMed. You need around 500GB or more free space for pretraining preprocessing and data generation. 
The configuration parameters are in `pmc_config_v2.json` file. The pretraining takes 3 weeks on a RTX 2070 8GB GPU.
An example pretraining configuration is in `pmc_config_v2.json.example` file. Please copy this file to `pmc_config_v2.json` and adjust the full paths according to your system's directory structure. The pretraining takes 3 weeks on a RTX 2070 8GB GPU.
```
cd $BIO_ELECTRA_HOME/electra
./build_pmc_oai_full_pretrain_dataset.sh
./pretrain_pmc_model_v2.sh
```

## Datasets
All of the datasets are available at `$BIO_ELECTRA_HOME/electra/data/finetuning_data`. 


## ELECTRA/Bio-ELECTRA biomedical text mining experiments
### Biomedical QA training/evaluation
```bash
train_bioasq_qa_baseline.sh # ELECTRA-Small++
train_bioasq_qa_pmc_1_8M.sh  # Bio-ELECTRA
train_bioasq_qa_pmc_v2_3_6M.sh  # Bio-ELECTRA++
```
#### Evaluation
After training, the results are stored under 
`$BIO_ELECTRA_HOME/electra/data/models/electra_small/results/`, `$BIO_ELECTRA_HOME/electra/data/models/pmc_electra_small_1_8_M/results/` and
`$BIO_ELECTRA_HOME/electra/data/models/pmc_electra_small_v2_3_6_M/results/` 
for Electra-Small++, Bio-ELECTRA and Bio-ELECTRA++, respectively.

For Bio-ELECTRA, copy the evaluation result files  `$BIO_ELECTRA_HOME/electra/data/models/pmc_electra_small_1_8_M/results/bioasq_results.txt`
and `$BIO_ELECTRA_HOME/electra/data/models/pmc_electra_small_1_8_M/results/bioasq_results.pkl` to 
`$BIO_ELECTRA_HOME/electra/pmc_results/qa_factoid/pmc_1_8M` directory.
Similarly, copy corresponding files for Electra-Small++ and Bio-ELECTRA++ from `$BIO_ELECTRA_HOME/electra/data/models` directory to 
`$BIO_ELECTRA_HOME/electra/pmc_results/qa_factoid/baseline` and `BIO_ELECTRA_HOME/electra/pmc_results/qa_factoid/pmc_v2_3_6M`, respectively.


Assuming the results are stored under `$BIO_ELECTRA_HOME/electra/pmc_results/qa_factoid`
```
python show_qa_performance.py --mode baseline # ELECTRA-Small++
python show_qa_performance.py --mode bio-electra
python show_qa_performance.py --mode bio-electra++

```

### Yes/No Question Classification training/evaluation
The yes/no question classification training/testing data is available at `$BIO_ELECTRA_HOME/electra/data/finetuning_data/yesno`. This dataset has no development set.

```
train_yesno_baseline.sh # ELECTRA-Small++
train_yesno.sh # Bio-ELECTRA
train_yesno_v2_3_6M.sh # Bio-ELECTRA++

```
#### Evaluation
After training, the results are stored under 
`$BIO_ELECTRA_HOME/electra/data/models/electra_small/results/`, `$BIO_ELECTRA_HOME/electra/data/models/pmc_electra_small_1_8_M/results/` and
`$BIO_ELECTRA_HOME/electra/data/models/pmc_electra_small_v2_3_6_M/results/` for Electra-Small++, Bio-ELECTRA and Bio-ELECTRA++, respectively.

For Bio-ELECTRA, copy the evaluation result files  `$BIO_ELECTRA_HOME/electra/data/models/pmc_electra_small_1_8_M/results/yesno_results.txt`
and `$BIO_ELECTRA_HOME/electra/data/models/pmc_electra_small_1_8_M/results/yesno_results.pkl` to 
`$BIO_ELECTRA_HOME/electra/pmc_results/yesno/pmc_1_8M` directory.
Similarly, copy corresponding files for Electra-Small++ and Bio-ELECTRA++ from `$BIO_ELECTRA_HOME/electra/data/models` directory to 
`$BIO_ELECTRA_HOME/electra/pmc_results/yesno/baseline` and `BIO_ELECTRA_HOME/electra/pmc_results/yesno/pmc_v2_3_6M`, respectively.


Assuming the results are stored under `$BIO_ELECTRA_HOME/electra/pmc_results/yes_no`
the following will show Bio-ELECTRA, ELECTRA-Small++ and Bio-ELECTRA++ test results;
```
python yesno_perf_stats.py
```

### Reranker training/evaluation
The reranker training/testing data is available at `$BIO_ELECTRA_HOME/data/bioasq_reranker`. This dataset is annotated by a single annotator and 
has no developement set.

```bash
./train_reranker_baseline.sh # ELECTRA-Small++
./train_reranker.sh # Bio-ELECTRA
./train_reranker_v2_3_6M.sh # Bio-ELECTRA++
```

#### Training with weighted objective function
```bash
./train_weighted_reranker_baseline.sh # ELECTRA-Small++ 
./train_weighted_reranker.sh # Bio-ELECTRA
./train_weighted_reranker_v2_3_6M.sh # Bio-ELECTRA++
```

#### Evaluation

```bash
./predict_reranker_baseline.sh # ELECTRA-Small++
./predict_reranker.sh # Bio-ELECTRA
./predict_reranker_v2_3_6M.sh
```

#### Prediction using weighted reranking models
```bash
./predict_weighted_reranker_baseline.sh
./predict_weighted_reranker.sh
./predict_weighted_reranker_v2_3_6M.sh
```

```
python show_reranker_performance.py --mode baseline
python show_reranker_performance.py --mode bio-electra
python show_reranker_performance.py --mode bio-electra++
python show_reranker_performance.py --mode weighted-baseline
python show_reranker_performance.py --mode weighted-bio-electra
python show_reranker_performance.py --mode weightede-bio-electra++
```

### Relation extraction training/evaluation
```bash
./train_re_gad_baseline.sh  # ELECTRA-Small++
./train_re_gad.sh # Bio-ELECTRA
./train_re_gad_v2_3_6M.sh # Bio-ELECTRA++
./train_re_chemprot_baseline.sh # ELECTRA-Small++
./train_re_chemprot.sh # Bio-ELECTRA
./train_re_chemprot_v2_3_6M.sh # Bio-ELECTRA++
```

#### Evaluation
```
python show_re_performance.py --mode gad-baseline
python show_re_performance.py --mode gad-bio-electra
python show_re_performance.py --mode gad-bio-electra++
python show_re_performance.py --mode chemprot-baseline
python show_re_performance.py --mode chemprot-bio-electra
python show_re_performance.py --mode chemprot-bio-electra++
```

### Biomedical named entity recognition training/evaluation
The datasets are located under the `$BIO_ELECTRA_HOME/electra/data/finetuning_data` directory.

```bash
./train_bc4chemd_ner_baseline.sh # ELECTRA-Small++
./train_bc4chemd_ner.sh # Bio-ELECTRA
./train_bc4chemd_ner_v2_3_6M.sh # Bio-ELECTRA++
./train_bc2gm_ner_baseline.sh # ELECTRA-Small++
./train_bc2gm_ner.sh # Bio-ELECTRA
./train_bc2gm_ner_v2_3_6M.sh # Bio-ELECTRA++
./train_linnaeus_ner_baseline.sh # ELECTRA-Small++
./train_linnaeus_ner.sh # Bio-ELECTRA
./train_linnaeus_ner_v2_3_6M.sh # Bio-ELECTRA++
./train_ncbi_disease_ner_baseline.sh # ELECTRA-Small++
./train_ncbi_disease_ner.sh # Bio-ELECTRA
./train_ncbi_disease_ner_v2_3_6M.sh # Bio-ELECTRA++

```

#### Evaluation
After training, the results are stored under 
`$BIO_ELECTRA_HOME/electra/data/models/electra_small/results/`, `$BIO_ELECTRA_HOME/electra/data/models/pmc_electra_small_1_8_M/results/` and
`$BIO_ELECTRA_HOME/electra/data/models/pmc_electra_small_v2_3_6_M/results/` 
for Electra-Small++, Bio-ELECTRA and Bio-ELECTRA++, respectively.

For Bio-ELECTRA bc4chemd NER data set, copy the evaluation result files  
`$BIO_ELECTRA_HOME/electra/data/models/pmc_electra_small_1_8_M/results/bc4chemd_results.txt`
and `$BIO_ELECTRA_HOME/electra/data/models/pmc_electra_small_1_8_M/results/bc4chemd_results.txt` to 
`$BIO_ELECTRA_HOME/electra/pmc_results/ner/pmc_1_8M/bc4chemd` directory.
Similarly, copy corresponding files for Electra-Small++ and Bio-ELECTRA++ from `$BIO_ELECTRA_HOME/electra/data/models` directory to 
`$BIO_ELECTRA_HOME/electra/pmc_results/ner/baseline/bc4chemd` and `BIO_ELECTRA_HOME/electra/pmc_results/ner/pmc_v2_3_6M/bc4chemd`, respectively.
The other three NER datasets have the prefix `bc2gm`, `linnaeus` and `ncbi_disease`.


Assuming the results are stored under `$BIO_ELECTRA_HOME/electra/pmc_results/ner`
the following will show Bio-ELECTRA, ELECTRA-Small++ and Bio-ELECTRA++ test results;
```
cd $BIO_ELECTRA_HOME/electra
python ner_perf_stats.py
```


## BERT biomedical text mining experiments

### Biomedical QA training/evaluation
```
./test_qa_bert_batch.sh
./qa_bert_perf_extract.sh > /tmp/bert_qa_perf.txt
python show_bert_perf_stats.py
```

### Yes/No Question Classification training/evaluation

```
./train_yesno_qc_bert_batch.sh
./test_yesno_qc_bert_batch.sh 
python show_bert_yesno_perf_stats.py
```

### Reranker training/evaluation
```
./train_bert_reranker_batch.sh
python show_reranker_performance.py
```

### Relation extraction training/evaluation

```
./train_bio_re_gad_bert_batch.sh
./test_bio_re_gad_bert_batch.sh
python show_re_performance.py --mode gad

./train_bio_re_chemprot_bert_batch.sh
./test_bio_re_chemprot_bert_batch.sh
python  show_re_performance.py --mode chemprot
```

## BERT biomedical NER experiments using transformers Python package

The four NER datasets are located under `$BIO_ELECTRA_HOME/bert_ner/data` directory.

### Training/prediction

The  following scripts train ten randomly initialized models on the corresponding training sets and evaluate the models on their corresponding test sets.
```bash
cd $BIO_ELECTRA_HOME/bert_ner
./run_tf_BC4CHEMD_batch.sh 
./run_tf_BC2GM_batch.sh
./run_tf_linnaeus_batch.sh
./run_tf_NCBI_disease_batch.sh
```

### Evaluation
```bash
python perf_stats.py --mode bc4chemd
python perf_stats.py --mode bc2gm
python perf_stats.py --mode linnaeus
python perf_stats.py --mode ncbi-disease
```
