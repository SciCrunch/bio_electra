import os
import pickle
import numpy as np


BIO_ELECTRA_HOME = os.getenv('BIO_ELECTRA_HOME')


def load_results(pickle_file):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    em_list = []
    f1_list = []
    for run in data:
        em_list.append(float(run['bioasq']['exact_match']))
        f1_list.append(float(run['bioasq']['f1']))
    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list)
    em_mean = np.mean(em_list)
    em_std = np.std(em_list)
    tmpl = "{} - Exact match: {:.2f} ({:.2f}) F1: {:.2f} ({:.2f})"
    prefix, fname = os.path.split(pickle_file)
    print(tmpl.format(prefix, em_mean, em_std, f1_mean, f1_std))
    return {'em': (mean_confidence_interval(em_list)),
            'f1': (mean_confidence_interval(f1_list))}


def show_baseline_results():
    root_dir = BIO_ELECTRA_HOME + '/electra/pmc_results/qa_factoid'
    load_results(root_dir + '/baseline/bioasq_results.pkl')


def show_bioelectra_results():
    root_dir = BIO_ELECTRA_HOME + '/electra/pmc_results/qa_factoid'
    load_results(root_dir + '/pmc_1_8M/bioasq_results.pkl')


def show_bioelectra_v2_results():
    root_dir = BIO_ELECTRA_HOME + '/electra/pmc_results/qa_factoid'
    load_results(root_dir + '/pmc_v2_3_6M/bioasq_results.pkl')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Show Bio-ELECTRA/ELECTRA question answering test performance')
    parser.add_argument('--mode', choices=['baseline', 'bio-electra', 'bio-electra++'], required=True)
    args = parser.parse_args()
    if args.mode == 'bio-electra':
        show_bioelectra_results()
    elif args.mode == 'bio-electra++':
        show_bioelectra_v2_results()
    elif args.mode == 'baseline':
        show_baseline_results()

