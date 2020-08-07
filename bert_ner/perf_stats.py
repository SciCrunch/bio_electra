import os
import os.path
from collections import defaultdict
import numpy as np

BIO_ELECTRA_HOME = os.getenv('BIO_ELECTRA_HOME')


def load_results(report_file, results):
    with open(report_file) as f:
        for line in f:
            line = line.rstrip()
            tokens = line.split()
            # print(tokens)
            if len(tokens) == 6:
                heading = tokens[0].strip()
                if heading == 'micro' or heading == 'macro':
                    if heading in results:
                        m = results[heading]
                    else:
                        m = defaultdict(list)
                        results[heading] = m
                    m['p'].append(float(tokens[2]) * 100)
                    m['r'].append(float(tokens[3]) * 100)
                    m['f1'].append( float(tokens[4]) * 100)
    return results

def show_perf_stats(perf_map, prefix):
    p_mean = np.mean(perf_map['p'])
    r_mean = np.mean(perf_map['r'])
    f1_mean = np.mean(perf_map['f1'])
    p_std = np.std(perf_map['p'])
    r_std = np.std(perf_map['r'])
    f1_std = np.std(perf_map['f1'])
    tmpl = "{} - P:{:.2f} ({:.2f}) R:{:.2f} ({:.2f}) F1:{:.2f} ({:.2f})" 
    print(tmpl.format(prefix, p_mean, p_std, r_mean, r_std,
                      f1_mean, f1_std))


def show_perf(results):
    show_perf_stats(results['micro'], 'Micro Averaged')
    show_perf_stats(results['macro'], 'Macro Averaged')


def show_model_perf(root_dir):
    results = {}
    for i in range(10):
        run_dir = os.path.join(root_dir,'run_' + str(i))
        report_file = os.path.join(run_dir,'test_results.txt')
        load_results(report_file, results)
    show_perf(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Show BERT NER  test performance')
    parser.add_argument('--mode', choices=['bc4chemd', 'bc2gm', 'linnaeus', 'ncbi-disease' ], required=True)
    args = parser.parse_args()

    root_dir = BIO_ELECTRA_HOME + '/bert_ner/'
    if args.mode == 'bc4chemd':
        print("BERT BC4CHEMD")
        show_model_perf(root_dir + '/bc4chemd_models')
    elif args.mode == 'bc2gm':
        print("BERT BC2GM")
        show_model_perf(root_dir + '/bc2gm_models')
    elif args.mode == 'ncbi-disease':
        print("NCBI Disease")
        show_model_perf(root_dir + '/ncbi_disease_models')
    elif args.mode == 'linnaeus':
        print('linnaeus')
        show_model_perf(root_dir + '/linnaeus_models')


