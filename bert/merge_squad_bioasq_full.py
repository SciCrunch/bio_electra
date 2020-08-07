import json


squad_json_file = '/home/bozyurt/dev/python/bert/squad/train-v1.1.json'
bioasq_json_file = '/home/bozyurt/dev/python/bert/bioasq_qa/qa_train-v1.1.json'
bioasq_json_dev_file = '/home/bozyurt/dev/python/bert/bioasq_qa/qa_dev-v1.1.json'
combined_json_file = '/home/bozyurt/dev/python/bert/bioasq_qa/qa_train-combined-full.json'


with open(squad_json_file) as f:
    data = json.load(f)

with open(bioasq_json_file) as f:
    bio_data = json.load(f)

with open(bioasq_json_dev_file) as f:
    bio_dev_data = json.load(f)

bio_data['data'][0]['paragraphs'].extend(bio_dev_data['data'][0]['paragraphs'])

data['data'].append(bio_data['data'][0])

with open(combined_json_file, 'w') as f:
    json.dump(data, f, indent=2)
print("wrote {}.".format(combined_json_file))


