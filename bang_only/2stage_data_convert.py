"""
@author: Guo Shiguang
@software: PyCharm
@file: 2stage_data_convert.py
@time: 2022/3/31 14:25
"""
from sys import argv

from transformers import BertTokenizer, BertForMaskedLM


def bert_uncased_tokenize(fin, fout):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    for line in fin:
        word_pieces = tokenizer.tokenize(line.strip().replace(' ##', ''))
        new_line = " ".join(word_pieces)
        fout.write('{}\n'.format(new_line))


fsrc = argv[1]
fcase = argv[2]
fout = argv[3]

with open(fsrc) as f:
    src_str = f.readlines()

with open(fcase) as f:
    case_str = f.readlines()

converted = []
for src, case in zip(src_str, case_str):
    case_str = case.split()
    keyword = [case_str[i + 1] for i, x in enumerate(case_str[:-1]) if x == '<extra_id_5>']
    for k in keyword:
        src = src.replace(k, '<extra_id_2> ' + k + ' <extra_id_3>')
    converted.append(src)

with open(fout + '.pre', 'w') as f:
    for item in converted:
        f.write(item)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
new_tokens = ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>', '<extra_id_3>', '<extra_id_5>']
num_added_toks = tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

bert_uncased_tokenize(fout + '.pre', fout)
