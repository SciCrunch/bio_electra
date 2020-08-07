import os
import csv
import json
import pickle
import numpy as np
import sklearn.metrics


BIO_ELECTRA_HOME = os.getenv('BIO_ELECTRA_HOME')


def read_tsv(tsv_file):
    with open(tsv_file, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def softmax(pred):
    return np.exp(pred)/np.sum(np.exp(pred))

def calc_f1_electra_stats(test_data_tsv_file, pickle_file_tmpl, no_trials):
    f1_list, r_list, p_list = [], [], []
    for i in range(1, no_trials+1):
        test_data_lines = read_tsv(test_data_tsv_file)
        npfile = pickle_file_tmpl.format(i)
        results = calc_f1_electra(test_data_lines, npfile)
        f1_list.append(results['f1'])
        r_list.append(results['recall'])
        p_list.append(results['precision'])
    tmpl = "Average P: {:.2f} ({:.2f}) R: {:.2f} ({:.2f}) F1: {:.2f} ({:.2f})"
    print(tmpl.format(np.mean(p_list), np.std(p_list), np.mean(r_list),
                np.std(r_list), np.mean(f1_list), np.std(f1_list)))



def calc_f1_electra(test_data_lines, pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    test_data_lines.pop(0)
    pred_list = list()
    y_true_list = list()
    for i, test_line in enumerate(test_data_lines):
        pred = data[i]
        pred_list.append(np.argmax(softmax(pred)))
        y_true_list.append(int(test_line[2]))
    p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_list,
                                                              y_true=y_true_list)
    results = dict()
    results["f1"] = f[1] * 100.0
    results["recall"] = r[1] * 100.0
    results["precision"] = p[1] * 100.0
    results["specificity"] = r[0]
    return results


def calc_f1_micro_av_stats(test_data_tsv_file, pickle_file_tmpl, no_trials, pos_labels, label_2_int_mapper):
    f1_list, r_list, p_list = [], [], []
    for i in range(1, no_trials+1):
        test_data_lines = read_tsv(test_data_tsv_file)
        npfile = pickle_file_tmpl.format(i)
        results = calc_f1_micro_av(test_data_lines, npfile,pos_labels,
                                   label_2_int_mapper)
        f1_list.append(results['f1'])
        r_list.append(results['recall'])
        p_list.append(results['precision'])
    tmpl = "Average P: {:.2f} ({:.2f}) R: {:.2f} ({:.2f}) F1: {:.2f} ({:.2f})"
    print(tmpl.format(np.mean(p_list), np.std(p_list), np.mean(r_list),
                np.std(r_list), np.mean(f1_list), np.std(f1_list)))


def calc_f1_micro_av(test_data_lines, pickle_file, pos_labels, label_2_int_mapper):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    test_data_lines.pop(0)
    pred_list = list()
    y_true_list = list()
    for i, test_line in enumerate(test_data_lines):
        pred = data[i]
        pred_list.append(np.argmax(softmax(pred)))
        y_true_list.append(label_2_int_mapper[test_line[1]])
    p,r,f,_ = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_list,
                                                              y_true=y_true_list,
                                                              labels=pos_labels,
                                                              average="micro")
    results = dict()
    results["f1"] = f * 100.0
    results["recall"] = r * 100.0
    results["precision"] = p * 100.0
    return results



def show_bio_electra_perf():
    test_data_tsv_file = BIO_ELECTRA_HOME + '/electra/data/finetuning_data/gad/test.tsv'
    tmpl = BIO_ELECTRA_HOME + '/electra/data/models/pmc_electra_small_1_8_M/test_predictions/gad_test_{}_predictions.pkl'
    calc_f1_electra_stats(test_data_tsv_file, tmpl, 10)


def show_bio_electra_v2_perf():
    test_data_tsv_file = BIO_ELECTRA_HOME + '/electra/data/finetuning_data/gad/test.tsv'
    tmpl = BIO_ELECTRA_HOME + '/electra/data/models/pmc_electra_small_v2_3_6_M/test_predictions/gad_test_{}_predictions.pkl'
    calc_f1_electra_stats(test_data_tsv_file, tmpl, 10)


def show_bio_electra_baseline_perf():
    test_data_tsv_file = BIO_ELECTRA_HOME + '/electra/data/finetuning_data/gad/test.tsv'
    tmpl = BIO_ELECTRA_HOME + '/electra/data/models/electra_small/test_predictions/gad_test_{}_predictions.pkl'
    calc_f1_electra_stats(test_data_tsv_file, tmpl, 10)


def show_chemprot_bioelectra_perf():
    test_data_tsv_file = BIO_ELECTRA_HOME + '/electra/data/finetuning_data/chemprot/test.tsv'
    tmpl = BIO_ELECTRA_HOME + '/electra/data/models/pmc_electra_small_1_8_M/test_predictions/chemprot_test_{}_predictions.pkl'
    label_2_int_mapper = {'CPR:3': 0, 'CPR:4': 1, 'CPR:5':2, 'CPR:6': 3,
                          'CPR:9': 4, 'False': 5}
    pos_labels= [0, 1, 2, 3, 4]
    print('Chemprot -- Bio-ELECTRA')
    calc_f1_micro_av_stats(test_data_tsv_file, tmpl, 10, pos_labels, label_2_int_mapper)

def show_chemprot_electra_perf():
    test_data_tsv_file = BIO_ELECTRA_HOME + '/electra/data/finetuning_data/chemprot/test.tsv'
    tmpl = BIO_ELECTRA_HOME + '/electra/data/models/electra_small/test_predictions/chemprot_test_{}_predictions.pkl'
    label_2_int_mapper = {'CPR:3': 0, 'CPR:4': 1, 'CPR:5':2, 'CPR:6': 3,
                          'CPR:9': 4, 'False': 5}
    pos_labels= [0, 1, 2, 3, 4]
    print('Chemprot -- ELECTRA Small++')
    calc_f1_micro_av_stats(test_data_tsv_file, tmpl, 10, pos_labels, label_2_int_mapper)

def show_chemprot_bioelectra_v2_perf():
    test_data_tsv_file = BIO_ELECTRA_HOME + '/electra/data/finetuning_data/chemprot/test.tsv'
    tmpl = BIO_ELECTRA_HOME + '/electra/data/models/pmc_electra_small_v2_3_6_M/test_predictions/chemprot_test_{}_predictions.pkl'
    label_2_int_mapper = {'CPR:3': 0, 'CPR:4': 1, 'CPR:5':2, 'CPR:6': 3,
                          'CPR:9': 4, 'False': 5}
    pos_labels= [0, 1, 2, 3, 4]
    print('Chemprot -- Bio-ELECTRA++')
    calc_f1_micro_av_stats(test_data_tsv_file, tmpl, 10, pos_labels, label_2_int_mapper)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Show ELECTRA/Bio-ELECTRA relation extraction test performance')
    parser.add_argument('--mode', choices=['gad-baseline','gad-bio-electra',
            'gad-bio-electra++', 'chemprot-baseline', 'chemprot-bio-electra',
            'chemprot-bio-electra++'], required=True)
    args = parser.parse_args()
    if args.mode == 'gad-baseline':
        show_bio_electra_baseline_perf()
    elif args.mode == 'gad-bio-electra':
        show_bio_electra_perf()
    elif args.mode == 'gad-bio-electra++':
        show_bio_electra_v2_perf()
    elif args.mode == 'chemprot-baseline':
        show_chemprot_electra_perf()
    elif args.mode == 'chemprot-bio-electra':
        show_chemprot_bioelectra_perf()
    elif args.mode == 'chemprot-bio-electra++':
        show_chemprot_bioelectra_v2_perf()
