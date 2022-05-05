"""
@author: Guo Shiguang
@software: PyCharm
@file: my_utils.py
@time: 2022/3/20 2:41
"""
import json
from copy import deepcopy
from typing import List

from UIE_light.spotasoc_predict_parser import PredictParser, SpotAsocPredictParser


class RecordSchema:
    def __init__(self, type_list, role_list, type_role_dict):
        self.type_list = type_list
        self.role_list = role_list
        self.type_role_dict = type_role_dict

    def __repr__(self) -> str:
        return f"Type: {self.type_list}\n" \
               f"Role: {self.role_list}\n" \
               f"Map: {self.type_role_dict}"

    @staticmethod
    def get_empty_schema():
        return RecordSchema(type_list=list(), role_list=list(), type_role_dict=dict())

    @staticmethod
    def read_from_file(filename):
        lines = open(filename).readlines()
        type_list = json.loads(lines[0])
        role_list = json.loads(lines[1])
        type_role_dict = json.loads(lines[2])
        type_list = [item.lower() for item in type_list]
        role_list = [item.lower() for item in role_list]
        type_role_dict = [item.lower() for item in type_role_dict]
        return RecordSchema(type_list, role_list, type_role_dict)

    def write_to_file(self, filename):
        with open(filename, 'w') as output:
            output.write(json.dumps(self.type_list) + '\n')
            output.write(json.dumps(self.role_list) + '\n')
            output.write(json.dumps(self.type_role_dict) + '\n')


class Metric:
    """ Tuple Metric """

    def __init__(self, verbose=False, match_mode='normal'):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.
        self.verbose = verbose
        self.match_mode = match_mode
        assert self.match_mode in {'set', 'normal', 'multimatch'}

    def __repr__(self) -> str:
        return f"tp: {self.tp}, gold: {self.gold_num}, pred: {self.pred_num}"

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        return {prefix + 'tp': tp,
                prefix + 'gold': gold_num,
                prefix + 'pred': pred_num,
                prefix + 'P': p * 100,
                prefix + 'R': r * 100,
                prefix + 'F1': self.safe_div(2 * p * r, p + r) * 100
                }

    def count_instance(self, gold_list, pred_list):
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)
            self.tp += len(gold_list & pred_list)

        else:
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)

            if len(gold_list) > 0 and len(pred_list) > 0:
                # guarantee length same
                assert len(gold_list[0]) == len(pred_list[0])

            dup_gold_list = deepcopy(gold_list)
            for pred in pred_list:
                if pred in dup_gold_list:
                    self.tp += 1
                    if self.match_mode == 'normal':
                        # Each Gold Instance can be matched one time
                        dup_gold_list.remove(pred)

    def count_batch_instance(self, batch_gold_list, batch_pred_list):
        for gold_list, pred_list in zip(batch_gold_list, batch_pred_list):
            self.count_instance(gold_list=gold_list, pred_list=pred_list)


class RecordMetric(Metric):
    """ 不考虑不同 Role 之间的顺序，例如事件论元"""

    @staticmethod
    def is_equal(gold, pred):
        if gold['type'] != pred['type']:
            return False
        if gold['trigger'] != pred['trigger']:
            return False
        if len(gold['roles']) != len(pred['roles']):
            return False
        for gold_role, pred_role in zip(sorted(gold['roles']), sorted(pred['roles'])):
            if gold_role != pred_role:
                return False
        return True

    def count_instance(self, gold_list, pred_list):
        if self.match_mode == 'set':
            raise NotImplementedError(f'{self.__class__.__name__} do not support the match model `set`')

        if self.verbose:
            print("Gold:", gold_list)
            print("Pred:", pred_list)

        self.gold_num += len(gold_list)
        self.pred_num += len(pred_list)

        gold_indexes = list(range(len(gold_list)))
        non_found = [True] * len(gold_list)
        for pred in pred_list:
            for gold_index in gold_indexes:
                if non_found[gold_index] and self.is_equal(gold_list[gold_index], pred):
                    self.tp += 1
                    non_found[gold_index] = False
                    if self.match_mode == 'normal':
                        break


class OrderedRecordMetric(RecordMetric):
    """ 考虑不同 Role 之间的顺序，例如关系 """

    @staticmethod
    def is_equal(gold, pred):
        if gold['type'] != pred['type']:
            return False
        if gold['trigger'] != pred['trigger']:
            return False
        if len(gold['roles']) != len(pred['roles']):
            return False
        for gold_role, pred_role in zip(gold['roles'], pred['roles']):
            if gold_role != pred_role:
                return False
        return True


def eval_pred(predict_parser: PredictParser, gold_list, pred_list, text_list=None, raw_list=None):
    well_formed_list, counter = predict_parser.decode(
        gold_list, pred_list, text_list, raw_list
    )

    spot_metric = Metric()
    asoc_metric = Metric()
    record_metric = RecordMetric()
    ordered_record_metric = OrderedRecordMetric()

    for instance in well_formed_list:
        spot_metric.count_instance(instance['gold_event'], instance['pred_event'])
        asoc_metric.count_instance(instance['gold_role'], instance['pred_role'])
        record_metric.count_instance(instance['gold_record'], instance['pred_record'])
        ordered_record_metric.count_instance(instance['gold_record'], instance['pred_record'])

    spot_result = spot_metric.compute_f1(prefix='spot-')
    asoc_result = asoc_metric.compute_f1(prefix='asoc-')
    record_result = record_metric.compute_f1(prefix='record-')
    ordered_record_result = ordered_record_metric.compute_f1(prefix='ordered-record-')

    overall_f1 = spot_result.get('spot-F1', 0.) + asoc_result.get('asoc-F1', 0.)
    result = {'overall-F1': overall_f1}
    result.update(spot_result)
    result.update(asoc_result)
    result.update(record_result)
    result.update(ordered_record_result)
    result.update(counter)
    return result


def get_extract_metrics(pred_lns: List[str], tgt_lns: List[str], label_constraint: RecordSchema):
    predict_parser = SpotAsocPredictParser(label_constraint)
    return eval_pred(
        predict_parser=predict_parser,
        gold_list=tgt_lns,
        pred_list=pred_lns
    )


def post_process_nar(raw_output):
    result = []
    for line in raw_output:
        line = line.strip()
        words = line.split(' ')
        words_filtered = []
        for w in words:
            if '[SEP]' in w and len(words_filtered) != 0:
                break
            elif '[SEP]' not in w and '[CLS]' not in w:
                words_filtered.append(w)
        line = ' '.join(words_filtered).replace(' ##', '')
        words = line.split(' ')
        words_filtered = []
        for w in words:
            if '[SEP]' in w and len(words_filtered) != 0:
                break
            elif '[SEP]' not in w and '[CLS]' not in w:
                words_filtered.append(w)
        result.append(' '.join(words_filtered))
    return result
