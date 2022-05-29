"""
@author: Guo Shiguang
@software: PyCharm
@file: EventCriterion.py
@time: 2022/4/26 20:36
"""
import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import Tensor

from UIE_light.UIELightGenerator import UIELightGenerator
from UIE_light.my_utils import RecordSchema, post_process_nar, get_extract_metrics


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('uie_light_loss')
class UIELightLoss(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        self.nar_ratio = args.nar_ratio
        self.schema = RecordSchema.read_from_file(args.schema_path)
        self.generator = UIELightGenerator(
            self.task.target_dictionary,
            beam_size=0,
            max_len_a=0,
            max_len_b=200,
            min_len=1,
            normalize_scores=True,
            len_penalty=1,
            unk_penalty=0,
            sampling=False,
            sampling_topk=-1,
            sampling_topp=-1.0,
            temperature=1.,
            diverse_beam_groups=-1,
            diverse_beam_strength=0.5,
            match_source_len=False,
            no_repeat_ngram_size=0,
            nar_max_length=-1,
        )

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        parser.add_argument('--nar-ratio', default=0., type=float, metavar='D',
                            help='0: AR, 1: NAR, ')
        parser.add_argument('--schema_path')
        # fmt: on

    def forward(self, model, sample, val=False, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # AR or NAR
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )

        results = model(src_tokens=src_tokens, src_lengths=src_lengths, empty_tokens=sample['empty_tokens'],
                        event_only_tokens=sample['event_only_tokens'],
                        include_arguments_tokens=sample['target'])

        losses, nll_loss = [], []
        for stage, outputs in results.items():
            for obj in outputs:
                _losses = self.compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=stage + '-' + obj + '-loss',
                )
                losses += [_losses]
                if outputs[obj].get("nll_loss", False):
                    nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 \
            else loss.new_tensor(0)

        if val:
            hypos = self.generator.generate([model], sample)

            src_dict = self.task.source_dictionary

            tgt_list = []
            hypo_str_list = []

            for i, sample_id in enumerate(sample['id'].tolist()):

                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], self.task.tgt_dict.pad())
                target_tokens = utils.strip_pad(sample['target'][i, :], self.task.tgt_dict.pad()).int().cpu()

                src_str = src_dict.string(src_tokens, None)
                target_str = self.task.tgt_dict.string(target_tokens, None, escape_unk=True)

                tgt_list.append(target_str)

                # Process top predictions
                for j, hypo in enumerate(hypos[i][:1]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'],
                        align_dict=None,
                        tgt_dict=self.task.target_dictionary,
                        remove_bpe=None,
                    )

                hypo_str_list.append(hypo_str)

            hypo_str_list = post_process_nar(hypo_str_list)
            results = get_extract_metrics(hypo_str_list, tgt_list, self.schema)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            'f1': results['overall-F1'] if val else 0,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data) if reduce else l[["loss"]].data
            )

        return loss, sample_size, logging_output

    def compute_loss(self, outputs, targets, masks, label_smoothing, name):
        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs_masked, targets_masked = outputs[masks], targets[masks]
        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            log_outputs = F.log_softmax(outputs_masked, dim=-1)
            losses = F.nll_loss(log_outputs, targets_masked.to(log_outputs.device), reduction='none')

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = nll_loss * (1 - label_smoothing) - mean_ds(log_outputs) * label_smoothing
            else:
                loss = nll_loss
        loss_dict = {"name": name, "loss": loss, "nll_loss": nll_loss}
        # if loss.data == 0:
        #     print('~~~~~')
        # print(loss_dict)
        return loss_dict

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        f1 = sum(log.get('f1', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'f1': f1
        }
        return agg_output
