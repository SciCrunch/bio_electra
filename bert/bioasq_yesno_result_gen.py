import json
import csv
import argparse
from collections import defaultdict


def prep_results_json(in_tsv_file, results_tsv_file):
    with open(in_tsv_file, 'r') as f:
        in_list = list(csv.reader(f, delimiter='\t'))
    with open(results_tsv_file, 'r') as f:
        result_list = list(csv.reader(f, delimiter='\t'))
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
    parser.add_argument('-r', action='store', help="classifier-results-tsv-file", required=True)
    parser.add_argument('-o', action='store', help="output-json-file", required=True)
    args = parser.parse_args()

    in_tsv_file = '/home/bozyurt/dev/java/bio-answerfinder/data/bioasq/yesno_classifier/yesno_test.tsv'
    results_tsv_file ='/tmp/yesno_predict_out/test_results.tsv'
    out_json_file = '/tmp/yesno_results.json'
    in_tsv_file = args.i
    results_tsv_file = args.r
    out_json_file = args.o

    r = prep_results_json(in_tsv_file, results_tsv_file)
    with open(out_json_file, 'w') as f:
        json.dump(r, f, indent=2)
    print("wrote {}.".format(out_json_file))

