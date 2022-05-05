"""
@author: Guo Shiguang
@software: PyCharm
@file: eval_loss_f1.py
@time: 2022/3/30 15:01
"""
import jsonlines
from tqdm import tqdm

from evaluate import evaluate

schemapath = '/data/guoshiguang/datasets/dyiepp_ace2005_subtype_converted/raw/event.schema'
expr_arch = 'ace2005-full-nar-f1'
suffix = ''

dirpath = "/data/guoshiguang/outputs/bang/ace2005-full-nar-150/150_epochs"

file = jsonlines.open(dirpath + 'results.jsonl', 'w')
for i in tqdm(range(110, 155, 5)):
    filepath = '{dirpath}/{i}'.format(dirpath=dirpath, i=i)
    r = evaluate(schemapath=schemapath, filepath=filepath)
    jsonlines.Writer.write(file, r)

file.close()
