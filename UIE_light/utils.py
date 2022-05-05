"""
@author: Guo Shiguang
@software: PyCharm
@file: utils.py
@time: 2022/3/19 13:48
"""

import os

import jsonlines


def get_index(lst, item):
    return [index for (index, value) in enumerate(lst) if value == item]


def get_tree(tree):
    depth = 0
    new_tree = []
    for item in tree.split():

        if item == '<extra_id_0>':
            depth += 1
        if depth <= 2:
            new_tree.append(item)
        if item == '<extra_id_1>':
            depth -= 1

    return ' '.join(new_tree)


def convert_tree(datapath, converted_path):
    if not os.path.exists(converted_path):
        os.mkdir(converted_path)
    split = ['train', 'val', 'test']
    for s in split:
        text = []
        case = []
        cased_txt = []
        event = []
        with open(os.path.join(datapath, s + '.json')) as f:
            for item in jsonlines.Reader(f):
                text.append(item['text'])
                event_list = item['event'].split()
                indices = get_index(event_list, '<extra_id_0>')
                for index in indices[1:][::-1]:
                    event_list.insert(index + 2, '<extra_id_5>')
                event_str = (' '.join(event_list))
                event.append(event_str)
                case_str = get_tree(event_str)
                case.append(case_str)
                case_str_list = case_str.split()
                keyword = [case_str_list[i + 1] for i, x in enumerate(case_str_list) if x == '<extra_id_5>']
                cased_txt_str = item['text']
                for k in keyword:
                    cased_txt_str = cased_txt_str.replace(k, '<extra_id_2> ' + k + ' <extra_id_3>')
                cased_txt.append(cased_txt_str)

        with open(os.path.join(converted_path, s + '.text'), 'w') as f:
            for item in text:
                f.write(item)
                f.write('\n')
        with open(os.path.join(converted_path, s + '.case'), 'w') as f:
            for item in case:
                f.write(item)
                f.write('\n')
        with open(os.path.join(converted_path, s + '.cased_txt'), 'w') as f:
            for item in cased_txt:
                f.write(item)
                f.write('\n')
        with open(os.path.join(converted_path, s + '.event'), 'w') as f:
            for item in event:
                f.write(item)
                f.write('\n')


convert_tree('../data/text2tree/dyiepp_ace2005_subtype', '../data/text2tree/dyiepp_ace2005_subtype_converted')
