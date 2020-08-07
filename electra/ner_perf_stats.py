import os
import pickle
import numpy as np


BIO_ELECTRA_HOME = os.getenv('BIO_ELECTRA_HOME')

def load_results(pickle_file, ner_task):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    p_list = []
    r_list = []
    f1_list = []
    for run in data:
        p_list.append(float(run[ner_task]['precision']))
        r_list.append(float(run[ner_task]['recall']))
        f1_list.append(float(run[ner_task]['f1']))
    tmpl = "{} - P: {:.2f} ({:.2f}) R: {:.2f} ({:.2f}) F1: {:.2f} ({:.2f})"
    pm = np.mean(p_list)
    rm = np.mean(r_list)
    f1m = np.mean(f1_list)
    p_std = np.std(p_list)
    r_std = np.std(r_list)
    f1_std = np.std(f1_list)
    print(tmpl.format(ner_task, pm, p_std, rm, r_std, f1m, f1_std))
    return {'p': p_list, 'r': r_list, 'f1': f1_list}


def main():
    print("Bio-ELECTRA\n----------")
    root_dir = BIO_ELECTRA_HOME + '/electra/pmc_results/ner/pmc_1_8M'
    load_results(root_dir + '/bc4chemd/bc4chemd_results.pkl','bc4chemd')
    load_results(root_dir + '/bc2gm/bc2gm_results.pkl','bc2gm')
    load_results(root_dir + '/ncbi_disease/ncbi-disease_results.pkl','ncbi-disease')
    load_results(root_dir + '/linnaeus/linnaeus_results.pkl','linnaeus')

    print("Baseline\n-------------")
    root_dir = BIO_ELECTRA_HOME +  '/electra/pmc_results/ner/baseline'
    load_results(root_dir + '/bc4chemd/bc4chemd_results.pkl','bc4chemd')
    load_results(root_dir + '/bc2gm/bc2gm_results.pkl','bc2gm')
    load_results(root_dir + '/ncbi_disease/ncbi-disease_results.pkl','ncbi-disease')
    load_results(root_dir + '/linnaeus/linnaeus_results.pkl','linnaeus')
    print("Bio-ELECTRA++\n-------------")
    root_dir = BIO_ELECTRA_HOME + '/electra/pmc_results/ner/pmc_v2_3_6M'
    load_results(root_dir + '/bc4chemd/bc4chemd_results.pkl','bc4chemd')
    load_results(root_dir + '/bc2gm/bc2gm_results.pkl','bc2gm')
    load_results(root_dir + '/ncbi_disease/ncbi-disease_results.pkl','ncbi-disease')
    load_results(root_dir + '/linnaeus/linnaeus_results.pkl','linnaeus')
    print("Bio-ELECTRA++ (opt)\n-------------")
    root_dir = BIO_ELECTRA_HOME + '/electra/pmc_results/ner/pmc_v2_3_6M_opt'
    load_results(root_dir + '/bc4chemd/bc4chemd_results.pkl','bc4chemd')
    load_results(root_dir + '/bc2gm/bc2gm_results.pkl','bc2gm')
    load_results(root_dir + '/ncbi_disease/ncbi-disease_results.pkl','ncbi-disease')
    load_results(root_dir + '/linnaeus/linnaeus_results.pkl','linnaeus')


if __main__ == "__main__":
    main()



