import os
import pickle
import numpy as np


BIO_ELECTRA_HOME = os.getenv('BIO_ELECTRA_HOME')


def load_results(pickle_file, task):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    p_yes_list = []
    r_yes_list = []
    f1_yes_list = []
    p_no_list = []
    r_no_list = []
    f1_no_list = []
    for run in data:
        p_yes_list.append(float(run[task]['precision_yes']))
        r_yes_list.append(float(run[task]['recall_yes']))
        f1_yes_list.append(float(run[task]['f1_yes']))
        p_no_list.append(float(run[task]['precision_no']))
        r_no_list.append(float(run[task]['recall_no']))
        f1_no_list.append(float(run[task]['f1_no']))
    tmpl = "{} [{}] - P: {:.2f} ({:.2f}) R: {:.2f} ({:.2f}) F1: {:.2f} ({:.2f})"
    pym = np.mean(p_yes_list)
    rym = np.mean(r_yes_list)
    f1ym = np.mean(f1_yes_list)
    py_std = np.std(p_yes_list)
    ry_std = np.std(r_yes_list)
    f1y_std = np.std(f1_yes_list)
    pnm = np.mean(p_no_list)
    rnm = np.mean(r_no_list)
    f1nm = np.mean(f1_no_list)
    pn_std = np.std(p_no_list)
    rn_std = np.std(r_no_list)
    f1n_std = np.std(f1_no_list)
    print(tmpl.format(task, 'Yes', pym, py_std, rym, ry_std, f1ym, f1y_std))
    print(tmpl.format(task, 'No', pnm, pn_std, rnm, rn_std, f1nm, f1n_std))


def main():
    root_dir = BIO_ELECTRA_HOME + '/electra/pmc_results/yesno/pmc_1_8M'
    print("Bio-Electra pmc_1_8M")
    load_results(root_dir + '/yesno_results.pkl', 'yesno')
    print("Baseline\n-------------")
    root_dir = BIO_ELECTRA_HOME + '/electra/pmc_results/yesno/baseline'
    load_results(root_dir + '/yesno_results.pkl', 'yesno')
    print("Bio-Electra++\n------------")
    root_dir = BIO_ELECTRA_HOME + '/electra/pmc_results/yesno/pmc_v2_3_6M'
    load_results(root_dir + '/yesno_results.pkl', 'yesno')


if __name__ == '__main__':
    main()
