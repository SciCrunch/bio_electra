import json
import numpy as np


def load_results(report_file, prefix):
    em_list = []
    f1_list = []
    with open(report_file) as f:
        for line in f:
            line = line.rstrip()
            data = json.loads(line)
            em_list.append(float(data['exact_match']))
            f1_list.append(float(data['f1']))
    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list)
    em_mean = np.mean(em_list)
    em_std = np.std(em_list)
    tmpl = "{} - Exact match: {:.2f} ({:.2f}) F1: {:.2f} ({:.2f})"
    print(tmpl.format(prefix, em_mean, em_std, f1_mean, f1_std))


if __name__ == "__main__":
    load_results('/tmp/bert_qa_perf.txt', 'bert_qa')
