import os
import csv
import json
import numpy as np
import pickle
import re

BIO_ELECTRA_HOME = os.getenv('BIO_ELECTRA_HOME')

class Candidate:
    def __init__(self, guid, sentence, label, score):
        self.guid = guid
        self.sentence = sentence
        self.label = label
        self.score = score


class QuestionAnswer:
    def __init__(self, question):
        self.question = question
        self.candidates = list()
        self.dict = {"question": self.question,
                    "answers":[]}

    def add_candidate(self, candidate):
        self.candidates.append(candidate)

    def sort(self):
        self.candidates.sort(key=lambda candidate: candidate.score,
                             reverse=True)

    def get_rank(self):
        """gets rank of the first correct answer (Call after sort)"""
        for i, candidate in enumerate(self.candidates):
            ans = {'ans': candidate.sentence, 'order': i + 1,
                   'score': candidate.score, 'gsa': False}
            if candidate.label == 1:
                ans['gsa'] = True
            self.dict['answers'].append(ans)
        for i, candidate in enumerate(self.candidates):
            if i == 0:
                print("Max Q:{}\nA:{}".format(self.question,
                        candidate.sentence))
            if candidate.label == 1:
                print("score:{} rank:{}".format(candidate.score, i + 1))
                print("Q:{}\nA:{}".format(self.question,
                        candidate.sentence))
                print("-------------------")
                return i + 1
        return 0


def read_tsv(tsv_file):
    with open(tsv_file, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def softmax(pred):
    return np.exp(pred)/np.sum(np.exp(pred))


def calc_mrr_electra_stats(test_data_tsv_file, pickle_file_tmpl, no_trials):
    mrr_list = []
    for i in range(1, no_trials+1):
        test_data_lines = read_tsv(test_data_tsv_file)
        npfile = pickle_file_tmpl.format(i)
        mrr = calc_mrr_electra(test_data_lines, npfile)
        mrr_list.append(mrr)
    tmpl = "Average MRR: {:.3f} ({:.3f})"
    print(tmpl.format(np.mean(mrr_list), np.std(mrr_list)))



def calc_mrr_electra(test_data_lines, pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    test_data_lines.pop(0)
    sum = 0.0
    qa_list = list()
    cur_qa = None
    for i, test_line in enumerate(test_data_lines):
        pred = data[i]
        class_1_prob = softmax(pred)[1]
        question = test_line[1]
        answer_sentence = test_line[2]
        label = int(test_line[0])
        quid = question
        if not cur_qa or cur_qa.question != question:
            cur_qa = QuestionAnswer(question)
            qa_list.append(cur_qa)
        cur_qa.add_candidate(Candidate(quid, answer_sentence, label,
                                       class_1_prob))
    print("# of questions:{}".format(len(qa_list)))
    dict_list = list()
    for qa in qa_list:
        qa.sort()
        rank = qa.get_rank()
        dict_list.append(qa.dict)
        if rank > 0:
            sum += 1.0 / rank
    mrr = sum / len(qa_list)
    print("MRR:{}".format(mrr))
    return mrr



def calc_mrr(test_data_lines, result_lines):
    test_data_lines.pop(0)
    sum = 0.0
    qa_list = list()
    cur_qa = None
    for (test_line, result_line) in zip(test_data_lines, result_lines):
        class_1_prob = float(result_line[1])
        label = int(result_line[3])
        guid = result_line[2]
        question = test_line[1]
        answer_sentence = test_line[2]
        if not cur_qa or cur_qa.question != question:
            cur_qa = QuestionAnswer(question)
            qa_list.append(cur_qa)
        cur_qa.add_candidate(Candidate(guid, answer_sentence, label,
                                       class_1_prob))
    print("# of questions:{}".format(len(qa_list)))
    dict_list = list()
    for qa in qa_list:
        qa.sort()
        rank = qa.get_rank()
        dict_list.append(qa.dict)
        if rank > 0:
            sum += 1.0 / rank
    mrr = sum / len(qa_list)
    print("MRR:{}".format(mrr))
    with open('/tmp/bert_perf_records.json', 'w') as f:
        json.dump(dict_list, f, indent=2)
    return mrr


def calc_performance(test_data_lines, result_lines):
    test_data_lines.pop(0)
    num_correct = 0
    num_answers = 0
    for (test_line, result_line) in zip(test_data_lines, result_lines):
        class_1_prob = float(result_line[1])
        label = int(result_line[3])
        if label == 1:
            if class_1_prob > 0.5:
                num_correct += 1
                print("* Q:{}\nA:{}".format(test_line[1], test_line[2]))
            else:
                print("Q:{}\nA:{}".format(test_line[1], test_line[2]))
            num_answers += 1
    acc = num_correct / float(num_answers)
    print("acc:{}".format(acc))
    return acc


def show_results(test_data_file, test_results_file):
    test_data_lines = read_tsv(test_data_file)
    result_lines = read_tsv(test_results_file)
    calc_mrr(test_data_lines, result_lines)


def show_electra_perf():
    test_data_tsv_file = BIO_ELECTRA_HOME + "/electra/data/finetuning_data/reranker/test.tsv"
    calc_mrr_electra_stats(test_data_tsv_file, BIO_ELECTRA_HOME + '/electra/data/models/pmc_electra_small_1_8_M/test_predictions/reranker_test_{}_predictions.pkl', 10)


def show_electra_baseline_perf():
    test_data_tsv_file = BIO_ELECTRA_HOME + "/electra/data/finetuning_data/reranker/test.tsv"
    calc_mrr_electra_stats(test_data_tsv_file, BIO_ELECTRA_HOME + '/electra/data/models/electra_small/test_predictions/reranker_test_{}_predictions.pkl', 10)


def show_electra_weighted_perf():
    test_data_tsv_file = BIO_ELECTRA_HOME + "/electra/data/finetuning_data/weighted-reranker/test.tsv"
    calc_mrr_electra_stats(test_data_tsv_file, BIO_ELECTRA_HOME + '/electra/data/models/pmc_electra_small_1_8_M/test_predictions/weighted-reranker_test_{}_predictions.pkl', 10)


def show_electra_weighted_baseline_perf():
    test_data_tsv_file = BIO_ELECTRA_HOME + "/electra/data/finetuning_data/weighted-reranker/test.tsv"
    calc_mrr_electra_stats(test_data_tsv_file, BIO_ELECTRA_HOME + '/electra/data/models/electra_small/test_predictions/weighted-reranker_test_{}_predictions.pkl', 10)


def show_bio_electra_v2_perf():
    print("Bio-ELECTRA++")
    test_data_tsv_file = BIO_ELECTRA_HOME + "/electra/data/finetuning_data/reranker/test.tsv"
    calc_mrr_electra_stats(test_data_tsv_file, BIO_ELECTRA_HOME + '/electra/data/models/pmc_electra_small_v2_3_6_M/test_predictions/reranker_test_{}_predictions.pkl', 10)


def show_bio_electra_v2_weighted_perf():
    print("Bio-ELECTRA++ (weighted)")
    test_data_tsv_file = BIO_ELECTRA_HOME + "/electra/data/finetuning_data/weighted-reranker/test.tsv"
    calc_mrr_electra_stats(test_data_tsv_file, BIO_ELECTRA_HOME + '/electra/data/models/pmc_electra_small_v2_3_6_M/test_predictions/weighted-reranker_test_{}_predictions.pkl', 10)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Show ELECTRA/Bio-ELECTRA reranking test performance')
    parser.add_argument('--mode', choices=['baseline',
            'bio-electra','bio_electra++','weighted-baseline',
            'weighted-bio-electra', 'weighted-bio-electra++'], required=True)
    args = parser.parse_args()
    if args.mode == 'baseline':
        show_electra_baseline_perf()
    elif args.mode == 'bio-electra':
        show_electra_perf()
    elif args.mode == 'bio_electra++':
        show_bio_electra_v2_perf()
    elif args.mode == 'weighted-baseline':
        show_electra_weighted_baseline_perf()
    elif args.mode == 'weighted-bio-electra':
        show_electra_weighted_perf()
    elif args.mode == 'weighted-bio-electra++':
        show_bio_electra_v2_weighted_perf()
