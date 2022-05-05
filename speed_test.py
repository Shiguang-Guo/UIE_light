"""
@author: Guo Shiguang
@software: PyCharm
@file: speed_test.py
@time: 2022/3/30 18:05
"""
import json
import random

from tqdm import tqdm

from pre_tokenize import bert_uncased_tokenize

random.seed(327)

material = "C:/Users/gsg18/Documents/本科毕设/speed_test/nyt.as"


def material_sampling():
    with open(material, 'r') as f:
        lines = f.readlines()
    sampled = random.sample(lines, 60000)
    slice_ = []
    for i in tqdm(sampled):
        if len(i) <= 500:
            slice_.append(i)
    with open('speed_test.txt', 'w') as f:
        for s in slice_[:50000]:
            f.write(s)


def text2event_format():
    new_format = []
    with open('speed_test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            new_format.append({"text": line[:-1], "event": ""})
    with open('speed_test_text2event_format.json', 'w') as f:
        for item in new_format:
            f.write(json.dumps(item))
            f.write('\n')


def bang_event():
    with open('speed_test.event', 'w') as f:
        f.write('<extra_id_0>\n' * 50000)

    bert_uncased_tokenize('speed_test.txt', 'speed_test_bang_format.txt')
    bert_uncased_tokenize('speed_test.event', 'speed_test_bang_format.event')

    # os.system('fairseq-preprocess \
    #  --user-dir /home/guoshiguang/bang/bang/bang \
    #  --task translation_bang \
    #  --source-lang txt --target-lang event \
    #  --testpref speed_test_bang_format \
    #  --destdir processed_data \
    #  --workers 20 --srcdict /home/guoshiguang/bang/bang/vocab.txt --tgtdict /home/guoshiguang/bang/bang/vocab.txt')


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# new_tokens = ['<extra_id_0>', '<extra_id_1>', '<extra_id_5>']
# num_added_toks = tokenizer.add_tokens(new_tokens)
# model.resize_token_embeddings(len(tokenizer))

# material_sampling()
text2event_format()
# bang_event()
