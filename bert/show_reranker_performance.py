import os
import csv
import json
import numpy as np


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

def calc_mrr_stats(test_data_tsv_file, result_file_tmpl, no_trials):
    mrr_list = []
    for i in range(no_trials):
        test_data_lines = read_tsv(test_data_tsv_file)
        result_file = result_file_tmpl.format(i)
        result_lines =  read_tsv(result_file)
        mrr = calc_mrr(test_data_lines, result_lines)
        mrr_list.append(mrr)
    tmpl = "Average MRR: {:.3f} ({:.3f})"
    print(tmpl.format(np.mean(mrr_list), np.std(mrr_list)))


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


def show_reranker_batch_perf():
    test_data_tsv_file = BIO_ELECTRA_HOME + "/electra/data/finetuning_data/reranker/test.tsv"
    result_data_tmpl = "/tmp/reranker_batch/run_{}/test_results.tsv"
    calc_mrr_stats(test_data_tsv_file, result_data_tmpl, 10)



if __name__ == "__main__":
    show_reranker_batch_perf()
