import csv
import json
import numpy as np


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

def calc_mrr_cv(cv_data_root_dir, cv_pred_root_dir):
    mrr_list = list()
    for i in range(1, 6):
        dev_file = cv_data_root_dir + "/fold_{}/dev.tsv".format(i)
        pred_file = cv_pred_root_dir + "/prediction_fold_{}/test_results.tsv".format(i)
        test_data_lines = read_tsv(dev_file)
        pred_lines = read_tsv(pred_file)
        mrr_list.append(calc_mrr(test_data_lines, pred_lines))
    print("MRR:{:.2f} ({:.2f})".format(np.mean(mrr_list),
                                       np.std(mrr_list)))

def calc_acc_cv(cv_data_root_dir, cv_pred_root_dir):
    acc_list = list()
    for i in range(1, 6):
        dev_file = cv_data_root_dir + "/fold_{}/dev.tsv".format(i)
        pred_file = cv_pred_root_dir + "/fold_{}/prediction/test_results.tsv".format(i)
        test_data_lines = read_tsv(dev_file)
        pred_lines = read_tsv(pred_file)
        acc_list.append(calc_performance(test_data_lines, pred_lines))
    print("accuracy:{:.2f} ({:.2f})".format(np.mean(acc_list),
                                       np.std(acc_list)))

def show_results(test_data_file, test_results_file):
    test_data_lines = read_tsv(test_data_file)
    result_lines = read_tsv(test_results_file)
    calc_mrr(test_data_lines, result_lines)


#test_data_lines = read_tsv(
#    "/home/bozyurt/dev/java/bnerkit/data/bioasq/bioasq_manual_100/dev.tsv")
#test_data_lines = read_tsv(
#    "/home/bozyurt/dev/java/bnerkit/data/bioasq/bioasq_manual_100/test.tsv")
#result_lines = read_tsv(
#    "/home/bozyurt/data/bert/bioasq/bioasq_predict_out/test_results.tsv")

#result_lines = read_tsv("/tmp/bioasq_predict_out/test_results.tsv")



#result_lines = read_tsv("/tmp/bioasq_predict_out_2/test_results.tsv")
#test_data_lines = read_tsv("/tmp/bioasq_manual_100_2/test.tsv")
# calc_performance(test_data_lines, result_lines)
#calc_mrr(test_data_lines, result_lines)

result_lines = read_tsv("/ws/models/bert/baseline/bioasq_predict_out/test_results.tsv")
test_data_lines = read_tsv(
    "/home/bozyurt/dev/java/bnerkit/data/bioasq/bioasq_manual_100/test.tsv")
calc_mrr(test_data_lines, result_lines)

print("Pubmed trained with domain specific vocabulary")

result_lines = read_tsv("/ws/models/bert/pmc_baseline/bioasq_pmc_predict_out/test_results.tsv")
test_data_lines = read_tsv(
    "/home/bozyurt/dev/java/bnerkit/data/bioasq/bioasq_manual_100/test.tsv")
calc_mrr(test_data_lines, result_lines)

print("bioasq_pmc_baseline_128")
print("-----------------------")
result_lines = read_tsv("/ws/models/bert/bioasq_pmc_baseline_128/bioasq_pmc_predict_out/test_results.tsv")
test_data_lines = read_tsv(
    "/home/bozyurt/dev/java/bnerkit/data/bioasq/bioasq_manual_100/test.tsv")
calc_mrr(test_data_lines, result_lines)

print("bioasq_pmc_baseline_128_2 + 400,000 steps (equivalent to total 1,600,000 steps")
print("--------------------------------------------------")
result_lines = read_tsv("/ws/models/bert/bioasq_pmc_baseline_128_2/bioasq_pmc_predict_out/test_results.tsv")
test_data_lines = read_tsv(
    "/home/bozyurt/dev/java/bnerkit/data/bioasq/bioasq_manual_100/test.tsv")
calc_mrr(test_data_lines, result_lines)

print("bioasq_pmc_baseline_128_3 + 400,000 steps (equivalent to total 2,400,000 steps")
print("--------------------------------------------------")
result_lines = read_tsv("/ws/models/bert/bioasq_pmc_baseline_128_3/bioasq_pmc_predict_out/test_results.tsv")
test_data_lines = read_tsv(
    "/home/bozyurt/dev/java/bnerkit/data/bioasq/bioasq_manual_100/test.tsv")
calc_mrr(test_data_lines, result_lines)


#calc_mrr_cv("/tmp/bert_cv", "/tmp/bioasq_output")
#calc_acc_cv("/tmp/bert_cv", "/tmp/bioasq_output")

#show_results('/tmp/bioasq_manual_100_3/test.tsv',
#             '/tmp/bioasq_multi_output/prediction_3/test_results.tsv')

