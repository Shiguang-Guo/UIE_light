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
        # print("***** Evaluate {setname} set *****".format(setname=split))
        golden_tokens = []
        pred_tokens = []
        # with open('/data/guoshiguang/outputs/bang/ace2005-full-ar/output_ar_pelt1.2_{0}_beam4.txt'.format(split)) as f:
        #     for line in f.readlines():
        #         if line.startswith('T-'):
        #             golden_tokens.append(' '.join(line.split()[1:]).replace(' ##', ''))
        #         if line.startswith('H-'):
        #             pred_tokens.append(' '.join(line.split()[2:]).replace(' ##', ''))

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
    filepath = "/data/guoshiguang/outputs/uie_light/ace2005-full-uie-light-v1/"

    print(evaluate(schemapath=schemapath, filepath=filepath))
