import json
import csv
import argparse
import pickle
import numpy as np
from collections import defaultdict

def load_results(results_pickle_file):
    with open(results_pickle_file, 'rb') as f:
        data = pickle.load(f)
    results = []
    for i in range(len(data)):
        probs = np.exp(data[i])/ np.sum(np.exp(data[i]))
        results.append( (probs[0], probs[1]) )
    return results


def prep_results_json(in_tsv_file, results_pickle_file):
    with open(in_tsv_file, 'r') as f:
        in_list = list(csv.reader(f, delimiter='\t'))
    result_list = load_results(results_pickle_file)
    in_list.pop(0)
    json_dict = defaultdict(list)
    for in_row, result_row in zip(in_list, result_list):
        question = in_row[1]
        sentence = in_row[2]
        no_prob = float(result_row[0])
        yes_prob = float(result_row[1])
        result = {'sentence': sentence, 'yes':yes_prob, 'no':no_prob}
        json_dict[question].append(result)
    return json_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', help="in-tsv-file", required=True)
    parser.add_argument('-r', action='store', help="classifier-results-pickle-file", required=True)
    parser.add_argument('-o', action='store', help="output-json-file", required=True)
    #args = parser.parse_args()

    in_tsv_file = '/home/bozyurt/dev/python/electra/data/finetuning_data/yesno/test.tsv'
    results_pickle_file ='/home/bozyurt/dev/python/electra/data/models/pmc_electra_small/test_predictions/yesno_test_1_predictions.pkl'
    out_json_file = '/tmp/yesno_results.json'
    #in_tsv_file = args.i
    #results_tsv_file = args.r
    #out_json_file = args.o

    r = prep_results_json(in_tsv_file, results_pickle_file)
    with open(out_json_file, 'w') as f:
        json.dump(r, f, indent=2)
    print("wrote {}.".format(out_json_file))

