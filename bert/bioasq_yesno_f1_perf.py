import json
import csv
import argparse
from collections import defaultdict


def prep_results_f1_perf(in_tsv_file, results_tsv_filei, out_file):
    with open(in_tsv_file, 'r') as f:
        in_list = list(csv.reader(f, delimiter='\t'))
    with open(results_tsv_file, 'r') as f:
        result_list = list(csv.reader(f, delimiter='\t'))
    in_list.pop(0)
    json_dict = defaultdict(list)
    n_no_correct, n_no_predicted, n_no_gold = 0, 0, 0
    n_yes_correct, n_yes_predicted, n_yes_gold = 0, 0, 0
    for in_row, result_row in zip(in_list, result_list):
        label = int(in_row[0])
        question = in_row[1]
        sentence = in_row[2]
        no_prob = float(result_row[0])
        yes_prob = float(result_row[1])
        if label == 1:
            n_yes_gold += 1
            if yes_prob > 0.5:
                n_yes_predicted += 1
                n_yes_correct += 1
            else:
                n_no_predicted += 1
        else:
            n_no_gold += 1
            if no_prob > 0.5:
                n_no_predicted += 1
                n_no_correct += 1
            else:
                n_yes_predicted += 1

    if n_yes_correct == 0:
      p_yes, r_yes, f1_yes = 0, 0, 0
    else:
      p_yes = 100.0 * n_yes_correct / n_yes_predicted
      r_yes = 100.0 * n_yes_correct / n_yes_gold
      f1_yes = 2 * p_yes * r_yes / (p_yes + r_yes)

    if n_no_correct == 0:
      p_no, r_no, f1_no = 0, 0, 0
    else:
      p_no = 100.0 * n_no_correct / n_no_predicted
      r_no = 100.0 * n_no_correct / n_no_gold
      f1_no = 2 * p_no * r_no / (p_no + r_no)

    print("Yes P:{} R:{} F1:{}".format(p_yes, r_yes, f1_yes))
    print("No P:{} R:{} F1:{}".format(p_no, r_no, f1_no))
    with open(out_file, "a+") as f:
        f.write("Yes,P:{},R:{},F1:{}\r\n".format(p_yes, r_yes, f1_yes))
        f.write("No,P:{},R:{},F1:{}\r\n".format(p_no, r_no, f1_no))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', help="in-tsv-file", required=True)
    parser.add_argument('-r', action='store', help="classifier-results-tsv-file", required=True)
    parser.add_argument('-o', action='store', help="output-f1-file", required=True)
    args = parser.parse_args()

    in_tsv_file = '/home/bozyurt/dev/java/bio-answerfinder/data/bioasq/yesno_classifier/yesno_test.tsv'
    results_tsv_file ='/tmp/yesno_predict_out/test_results.tsv'
    out_json_file = '/tmp/yesno_results.json'
    in_tsv_file = args.i
    results_tsv_file = args.r
    out_file = args.o

    r = prep_results_f1_perf(in_tsv_file, results_tsv_file, out_file)

