"""
@author: Guo Shiguang
@software: PyCharm
@file: evaluate.py
@time: 2022/3/9 13:02
"""

from UIE_light.my_utils import RecordSchema, get_extract_metrics


def evaluate(schemapath, filepath):
    split_set = ['train', 'valid', 'test']
    schema = RecordSchema.read_from_file(schemapath)
    result = {}
    for split in split_set:
        with open("{filepath}/{split}_hypo.txt".format(filepath=filepath, split=split)) as f:
            pred_tokens = f.readlines()
        with open("{filepath}/{split}_golden.txt".format(filepath=filepath, split=split)) as f:
            golden_tokens = f.readlines()

        result['{}_{}'.format(filepath.split('/')[-1], split)] = get_extract_metrics(pred_lns=pred_tokens,
                                                                                     tgt_lns=golden_tokens,
                                                                                     label_constraint=schema)
    return result


if __name__ == '__main__':
    schemapath = '/data/guoshiguang/datasets/dyiepp_ace2005_subtype_converted/raw/event.schema'
    filepath = "/data/guoshiguang/outputs/uie_light/ace2005-full-uie-light-noshare-distilled-1/"

    print(evaluate(schemapath=schemapath, filepath=filepath))
