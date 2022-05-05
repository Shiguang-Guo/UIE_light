"""
@author: Guo Shiguang
@software: PyCharm
@file: spotasoc_predict_parser.py
@time: 2022/3/20 2:44
"""

import logging
import os
import re
from collections import Counter
from typing import Dict
from typing import List, Counter, Tuple

from nltk.tree import ParentedTree


def fix_unk_from_text(span, text, unk='<unk>'):
    """
    Find span from the text to fix unk in the generated span
    从 text 中找到 span，修复span

    Example:
    span = "<unk> colo e Bengo"
    text = "At 159 meters above sea level , Angola International Airport is located at Ícolo e Bengo , part of Luanda Province , in Angola ."

    span = "<unk> colo e Bengo"
    text = "Ícolo e Bengo , part of Luanda Province , in Angola ."

    span = "Arr<unk> s negre"
    text = "The main ingredients of Arròs negre , which is from Spain , are white rice , cuttlefish or squid , cephalopod ink , cubanelle and cubanelle peppers . Arròs negre is from the Catalonia region ."

    span = "colo <unk>"
    text = "At 159 meters above sea level , Angola International Airport is located at e Bengo , part of Luanda Province , in Angola . coloÍ"

    span = "Tarō As<unk>"
    text = "The leader of Japan is Tarō Asō ."

    span = "Tar<unk> As<unk>"
    text = "The leader of Japan is Tarō Asō ."

    span = "<unk>Tar As<unk>"
    text = "The leader of Japan is ōTar Asō ."
    """
    if unk not in span:
        return span

    def clean_wildcard(x):
        return x.replace('(', r'\(').replace(')', r'\)').replace('[', r'\[').replace(']', r'\]')

    match = r'\s*\S\s*'.join([clean_wildcard(item.strip()) for item in span.split(unk)])
    result = re.search(match, text)
    if not result:
        return span
    return result.group().strip()


class PredictParser:
    def __init__(self, label_constraint=None):
        self.predicate_set = label_constraint.type_list if label_constraint else list()
        self.role_set = label_constraint.role_list if label_constraint else list()

    def decode(self, gold_list, pred_list, text_list=None, raw_list=None) -> Tuple[List, Counter]:
        """

        :param gold_list:
        :param pred_list:
        :param text_list:
        :param raw_list:
        :return:
            dict:
                pred_event -> [(type1, trigger1), (type2, trigger2), ...]
                gold_event -> [(type1, trigger1), (type2, trigger2), ...]
                pred_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
                gold_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
                pred_record -> [{'type': type1, 'trigger': trigger1, 'roles': [(type1, role1, argument1), ...]},
                                {'type': type2, 'trigger': trigger2, 'roles': [(type2, role2, argument2), ...]},
                                ]
                gold_record -> [{'type': type1, 'trigger': trigger1, 'roles': [(type1, role1, argument1), ...]},
                                {'type': type2, 'trigger': trigger2, 'roles': [(type2, role2, argument2), ...]},
                                ]
            Counter:
        """
        pass

    @staticmethod
    def count_multi_event_role_in_instance(instance, counter):
        if len(instance['gold_event']) != len(set(instance['gold_event'])):
            counter.update(['multi-same-event-gold'])

        if len(instance['gold_role']) != len(set(instance['gold_role'])):
            counter.update(['multi-same-role-gold'])

        if len(instance['pred_event']) != len(set(instance['pred_event'])):
            counter.update(['multi-same-event-pred'])

        if len(instance['pred_role']) != len(set(instance['pred_role'])):
            counter.update(['multi-same-role-pred'])


type_start = '<extra_id_0>'
type_end = '<extra_id_1>'
span_start = '<extra_id_5>'
null_span = '<extra_id_6>'

logger = logging.getLogger(__name__)

debug = True if 'DEBUG' in os.environ else False

left_bracket = '【'
right_bracket = '】'
brackets = left_bracket + right_bracket

split_bracket = re.compile(r"<extra_id_\d>")


def add_space(text):
    """
    add space between special token
    :param text:
    :return:
    """
    new_text_list = list()
    for item in zip(split_bracket.findall(text), split_bracket.split(text)[1:]):
        new_text_list += item
    return ' '.join(new_text_list)


def convert_bracket(text):
    text = add_space(text)
    for start in [type_start]:
        text = text.replace(start, left_bracket)
    for end in [type_end]:
        text = text.replace(end, right_bracket)
    return text


def find_bracket_num(tree_str):
    """
    Count Bracket Number, 0 indicate num_left = num_right
    :param tree_str:
    :return:
    """
    count = 0
    for char in tree_str:
        if char == left_bracket:
            count += 1
        elif char == right_bracket:
            count -= 1
        else:
            pass
    return count


def check_well_form(tree_str):
    return find_bracket_num(tree_str) == 0


def clean_text(tree_str):
    count = 0
    sum_count = 0

    tree_str_list = tree_str.split()

    for index, char in enumerate(tree_str_list):
        if char == left_bracket:
            count += 1
            sum_count += 1
        elif char == right_bracket:
            count -= 1
            sum_count += 1
        else:
            pass
        if count == 0 and sum_count > 0:
            return ' '.join(tree_str_list[:index + 1])
    return ' '.join(tree_str_list)


def resplit_label_span(label, span, split_symbol=span_start):
    label_span = label + ' ' + span

    if split_symbol in label_span:
        try:
            new_label, new_span = label_span.split(split_symbol)
            return new_label.strip(), new_span.strip()
        except:
            # print('resplit_label_span error:', label_span, split_symbol)

            pass

    return label, span


def add_bracket(tree_str):
    """
    add right bracket to fill ill-formed
    :param tree_str:
    :return:
    """
    tree_str_list = tree_str.split()
    bracket_num = find_bracket_num(tree_str_list)
    tree_str_list += [right_bracket] * bracket_num
    return ' '.join(tree_str_list)


def get_tree_str(tree):
    """
    get str from event tree
    :param tree:
    :return:
    """
    str_list = list()
    for element in tree:
        if isinstance(element, str):
            str_list += [element]
    return ' '.join(str_list)


def rewrite_label_span(label, span, label_set=None, text=None):
    # Invalid Type
    if label_set and label not in label_set:
        print('Invalid Label: %s' % label) if debug else None
        return None, None

    # Fix unk using Text
    if text is not None and '<unk>' in span:
        span = fix_unk_from_text(span, text, '<unk>')

    # Invalid Text Span
    if text is not None and span not in text:
        print('Invalid Text Span: %s\n%s\n' % (span, text)) \
            if debug else None
        return None, None

    return label, span


class SpotAsocPredictParser(PredictParser):

    def decode(self, gold_list, pred_list, text_list=None, raw_list=None
               ) -> Tuple[List[Dict], Counter]:
        """

        :param gold_list:
        :param pred_list:
        :param text_list:
        :param raw_list:
        :return:
            dict:
                pred_event -> [(type1, trigger1), (type2, trigger2), ...]
                gold_event -> [(type1, trigger1), (type2, trigger2), ...]
                pred_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
                gold_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
                pred_record -> [{'type': type1, 'trigger': trigger1, 'roles': [(type1, role1, argument1), ...]},
                                {'type': type2, 'trigger': trigger2, 'roles': [(type2, role2, argument2), ...]},
                                ]
                gold_record -> [{'type': type1, 'trigger': trigger1, 'roles': [(type1, role1, argument1), ...]},
                                {'type': type2, 'trigger': trigger2, 'roles': [(type2, role2, argument2), ...]},
                                ]
            Counter:
        """
        counter = Counter()
        well_formed_list = []

        if gold_list is None or len(gold_list) == 0:
            gold_list = ["%s%s" % (type_start, type_end)] * len(pred_list)

        if text_list is None:
            text_list = [None] * len(pred_list)

        if raw_list is None:
            raw_list = [None] * len(pred_list)

        for gold, pred, text, raw_data in zip(gold_list, pred_list, text_list, raw_list):
            gold = gold.replace(' - ', '-')
            pred = pred.replace(' - ', '-')
            gold = convert_bracket(gold)
            pred = convert_bracket(pred)

            pred = clean_text(pred)

            try:
                gold_tree = ParentedTree.fromstring(gold, brackets=brackets)
            except ValueError:
                # logger.warning(f"Ill gold: {gold}")
                # logger.warning(f"Fix gold: {add_bracket(gold)}")
                try:
                    gold_tree = ParentedTree.fromstring(add_bracket(gold), brackets=brackets)
                    counter.update(['gold_tree add_bracket'])
                except:
                    print(add_bracket(gold))

            instance = {'gold': gold,
                        'pred': pred,
                        'gold_tree': gold_tree,
                        'text': text,
                        'raw_data': raw_data
                        }

            counter.update(['gold_tree' for _ in gold_tree])

            instance['gold_event'], instance['gold_role'], instance['gold_record'] = self.get_record_list(
                tree=instance["gold_tree"],
                text=instance['text']
            )

            try:
                if not check_well_form(pred):
                    pred = add_bracket(pred)
                    counter.update(['fixed'])

                pred_tree = ParentedTree.fromstring(pred, brackets=brackets)
                counter.update(['pred_tree' for _ in pred_tree])

                instance['pred_tree'] = pred_tree
                counter.update(['well-formed'])

            except ValueError:
                counter.update(['ill-formed'])
                print('ill-formed', pred) if debug else None
                instance['pred_tree'] = ParentedTree.fromstring(
                    left_bracket + right_bracket,
                    brackets=brackets
                )

            instance['pred_event'], instance['pred_role'], instance['pred_record'] = self.get_record_list(
                tree=instance["pred_tree"],
                text=instance['text']
            )

            self.count_multi_event_role_in_instance(
                instance=instance, counter=counter)

            well_formed_list += [instance]

        return well_formed_list, counter

    def get_record_list(self, tree, text=None):

        spot_list = list()
        asoc_list = list()
        record_list = list()

        # if len(tree) < 2:
        #     return event_list, role_list, record_list
        # tree = tree[1]
        # if len(tree) < 1:
        #     return event_list, role_list, record_list

        for spot_tree in tree:

            if isinstance(spot_tree, str):
                continue

            if len(spot_tree) == 0:
                continue

            spot_type = spot_tree.label()
            spot_trigger = get_tree_str(spot_tree)
            spot_type, spot_trigger = resplit_label_span(
                spot_type, spot_trigger)
            spot_type, spot_trigger = rewrite_label_span(
                label=spot_type,
                span=spot_trigger,
                label_set=self.predicate_set,
                text=text
            )

            if spot_trigger == null_span:
                continue

            if spot_type is None or spot_trigger is None:
                continue

            record = {'roles': list(),
                      'type': spot_type,
                      'trigger': spot_trigger}

            for asoc_tree in spot_tree:
                if isinstance(asoc_tree, str) or len(asoc_tree) < 1:
                    continue

                asoc_label = asoc_tree.label()
                asoc_text = get_tree_str(asoc_tree)
                asoc_label, asoc_text = resplit_label_span(
                    asoc_label, asoc_text)
                asoc_label, asoc_text = rewrite_label_span(
                    label=asoc_label,
                    span=asoc_text,
                    label_set=self.role_set,
                    text=text
                )

                if asoc_text == null_span:
                    continue
                if asoc_label is None or asoc_text is None:
                    continue

                asoc_list += [(spot_type, asoc_label, asoc_text)]
                record['roles'] += [(asoc_label, asoc_text)]

            spot_list += [(spot_type, spot_trigger)]
            record_list += [record]

        return spot_list, asoc_list, record_list
