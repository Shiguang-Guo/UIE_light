"""
@author: Guo Shiguang
@software: PyCharm
@file: pre_tokenize.py
@time: 2022/3/20 15:10
"""
import os.path
from itertools import product

from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM


def bert_uncased_tokenize(fin, fout):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    for line in tqdm(fin):
        word_pieces = tokenizer.tokenize(line.strip().replace(' ##', ''))
        new_line = " ".join(word_pieces)
        fout.write('{}\n'.format(new_line))


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
new_tokens = ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>', '<extra_id_3>', '<extra_id_5>']
num_added_toks = tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

split = ['train', 'valid']
type = ['text', 'event']

fin_path = '/data/guoshiguang/datasets/distilled/raw'
fout_path = '/data/guoshiguang/datasets/distilled/tokenized'

for item in product(split, type):
    bert_uncased_tokenize(os.path.join(fin_path, '{}.{}'.format(item[0], item[1])),
                          os.path.join(fout_path, '{}.{}'.format(item[0], item[1])))
