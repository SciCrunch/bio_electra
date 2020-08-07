import os
import numpy as np

BIO_ELECTRA_HOME = os.getenv('BIO_ELECTRA_HOME')

def ev(token):
    tokens = token.split(':')
    assert len(tokens) == 2
    return tokens[1]


def load_results(results_file, task):
    data = []
    with open(results_file) as f:
        lines = f.readlines()
        for i in range(len(lines)//2):
            yes_tokens = lines[2*i].split(',')
            no_tokens = lines[2*i + 1].split(',')
            r = {'precision_yes': ev(yes_tokens[1]), 
                 'recall_yes': ev(yes_tokens[2]), 
                 'f1_yes': ev(yes_tokens[3]), 
                 'precision_no': ev(no_tokens[1]), 
                 'recall_no': ev(no_tokens[2]), 
                 'f1_no': ev(no_tokens[3])}
            data.append(r)

    p_yes_list = []
    r_yes_list = []
    f1_yes_list = []
    p_no_list = []
    r_no_list = []
    f1_no_list = []
    for run in data:
        p_yes_list.append(float(run['precision_yes']))
        r_yes_list.append(float(run['recall_yes']))
        f1_yes_list.append(float(run['f1_yes']))
        p_no_list.append(float(run['precision_no']))
        r_no_list.append(float(run['recall_no']))
        f1_no_list.append(float(run['f1_no']))
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


if __name__ == "__main__":
    load_results('/tmp/bert_yesno_f1_results.csv', 'yesno')
