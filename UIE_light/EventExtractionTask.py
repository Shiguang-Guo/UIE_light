"""
@author: Guo Shiguang
@software: PyCharm
@file: EventExtractionTask.py
@time: 2022/4/18 13:59
"""
import torch
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset

from UIE_light.EventDictionary import EventDictionary
from UIE_light.UIELightGenerator import UIELightGenerator


@register_task('EventExtraction')
class EventExtractionTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # TO
        paths = self.args.data.split()
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            prepend_bos=True,
        )

    @classmethod
    def load_dictionary(cls, filename):
        return EventDictionary.load_from_file(filename=filename)

    def preprocess(self, tokens):
        pad = self.tgt_dict.pad()
        span_bos = self.tgt_dict.span_bos()
        span_eos = self.tgt_dict.span_eos()
        bos = self.tgt_dict.bos()
        eos = self.tgt_dict.eos()

        max_len = tokens.size(1)
        tokens_list = [[t for t in s if t != pad] for s in tokens.tolist()]
        events_only_list = []
        for item in tokens_list:
            events_only = [span_bos]
            depth = 0
            for token in item[2:-2]:
                if token == span_bos and depth < 1:
                    events_only.append(token)
                    depth += 1
                elif token == span_bos and depth >= 1:
                    # if events_only[-1] != span_placeholder:
                    #     events_only.append(span_placeholder)
                    depth += 1
                elif token == span_eos and depth > 1:
                    depth -= 1
                elif token == span_eos and depth <= 1:
                    events_only.append(token)
                    depth -= 1
                elif depth <= 1:
                    events_only.append(token)
            events_only_list.append([bos] + events_only + [span_eos, eos])
            events_only_list_padded = [
                item + [pad for _ in range(max_len - len(item))] for item in events_only_list
            ]

        # empty = [
        #     [bos, event_placeholder, eos] for _ in events_only_list
        # ]
        empty = [
            [bos, eos] for _ in events_only_list
        ]
        empty_padded = [
            item + [pad for _ in range(max_len - len(item))] for item in empty
        ]
        events_only_tensor_padded = torch.tensor(events_only_list_padded, device=tokens.device)
        empty_tensor = torch.tensor(empty_padded, device=tokens.device)
        return empty_tensor, events_only_tensor_padded

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   ignore_grad=False):
        model.train()
        sample['empty_tokens'], sample['event_only_tokens'] = self.preprocess(sample['target'])
        loss, sample_size, logging_output = criterion(model, sample, val=False)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        sample['empty_tokens'], sample['event_only_tokens'] = self.preprocess(sample['target'])
        with torch.no_grad():
            sample['empty'], sample['event_only'] = self.preprocess(sample['target'])
            loss, sample_size, logging_output = criterion(model, sample, val=True)
        return loss, sample_size, logging_output

    def build_generator(self, args):
        return UIELightGenerator(
            self.target_dictionary,
            beam_size=getattr(args, 'beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            sampling=getattr(args, 'sampling', False),
            sampling_topk=getattr(args, 'sampling_topk', -1),
            sampling_topp=getattr(args, 'sampling_topp', -1.0),
            temperature=getattr(args, 'temperature', 1.),
            diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
            diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            nar_max_length=getattr(args, 'nar_max_length', -1),
        )
