import json
import argparse
from collections import defaultdict
from collections import OrderedDict
from collections import namedtuple

Prediction = namedtuple('Prediction', 'text probability')


class QAResult(object):
    def __init__(self, qid, question):
        self.qid = qid
        self.question = question
        self.sr_map = dict()

    def add_sentence_result(self, sr):
        self.sr_map[sr.qsid] = sr

    def to_json(self):
        json = {'qid': self.qid, 'question': self.question}
        sr_list = [x.to_json() for x in self.sr_map.values()]
        sr_list.sort(key=lambda x : int(x['id'][x['id'].index(':') + 1:]))
        json['sentence_preds'] =  sr_list
        return json


class SentenceResult(object):
    def __init__(self, qsid, context):
        self.qsid = qsid
        self.context = context
        self.predictions = list()

    def to_json(self):
        json = {'context': self.context, 'id': self.qsid}
        json['predictions'] = [{'answer':x.text, 'prob':x.probability} for x in self.predictions]
        return json



def gen_result_json(in_data, nbest_data):
    result = OrderedDict()
    paragraphs = in_data['data'][0]['paragraphs']
    for para in paragraphs:
        qa = para['qas'][0]
        qsid = qa['id']
        qid = qsid[0:qsid.index(':')]
        question = qa['question']
        if qid in result:
            qa_result = result[qid]
        else:
            qa_result = QAResult(qid, question)
            result[qid] = qa_result
        sr = SentenceResult(qsid, para['context'])
        qa_result.add_sentence_result(sr)

    for qsid in nbest_data.keys():
        qid = qsid[0:qsid.index(':')]
        qa_result = result[qid]
        raw_preds = nbest_data[qsid]
        preds = [Prediction(text=x['text'], probability=x['probability']) for x in raw_preds]
        qa_result.sr_map[qsid].predictions = preds

    rs_list = [x.to_json() for x in result.values()]
    return rs_list

def main(in_json_file, nbest_pred_json_file, result_json_file):
    with open(in_json_file) as f:
        in_data = json.load(f)
    with open(nbest_pred_json_file) as f:
        nbest_data = json.load(f)
    rs_list = gen_result_json(in_data, nbest_data)
    with open(result_json_file, 'w') as f:
        json.dump(rs_list, f, indent=2)
    print("wrote {}.".format(result_json_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store',
            help="in-json-file-for-prediction", required=True)
    parser.add_argument('-r', action='store', 
            help="nbest-classifier-results-json-file", required=True)
    parser.add_argument('-o', action='store', help="output-json-file", required=True)
    args = parser.parse_args()
    
    in_json_file = '/home/bozyurt/dev/python/bert/bioasq_qa/qa_dev-v1.1.json'
    nbest_pred_json_file = '/tmp/qa_8b_biobert_combined_base/nbest_predictions.json'
    result_json_file = '/tmp/qa_nbest_results.json'

    in_json_file = args.i
    nbest_pred_json_file = args.r
    result_json_file = args.o

    main(in_json_file, nbest_pred_json_file, result_json_file)






