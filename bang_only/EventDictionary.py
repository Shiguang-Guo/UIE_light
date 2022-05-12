"""
@author: Guo Shiguang
@software: PyCharm
@file: EventDictionary.py
@time: 2022/4/21 14:27
"""
from fairseq.data import Dictionary


class EventDictionary(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
            self,
            pad='<pad>',
            eos='</s>',
            unk='<unk>',
            bos='<s>',
            extra_special_symbols=None,
    ):
        super().__init__(pad=pad, eos=eos, unk=unk, bos=bos, extra_special_symbols=extra_special_symbols)
        self.span_bos_index = None
        self.span_eos_index = None
        self.span_delimiter_index = None
        self.span_placeholder_index = None
        self.event_placeholder_index = None

    @classmethod
    def load_from_file(cls, filename):
        d = cls()
        d.symbols = []
        d.count = []
        d.indices = {}

        with open(filename, 'r', encoding='utf-8', errors='ignore') as input_file:
            for line in input_file:
                k, v = line.split(' ')
                d.add_symbol(k)

        d.unk_word = '[UNK]'
        d.pad_word = '[PAD]'
        d.eos_word = '[SEP]'
        d.bos_word = '[CLS]'
        span_bos = '<extra_id_0>'
        span_eos = '<extra_id_1>'
        span_delimiter = '<extra_id_5>'
        span_placeholder = '<span_plh>'
        event_placeholder = '<event_plh>'

        d.bos_index = d.add_symbol('[CLS]')
        d.pad_index = d.add_symbol('[PAD]')
        d.eos_index = d.add_symbol('[SEP]')
        d.unk_index = d.add_symbol('[UNK]')
        d.span_bos_index = d.add_symbol(span_bos)
        d.span_eos_index = d.add_symbol(span_eos)
        d.span_delimiter_index = d.add_symbol(span_delimiter)
        d.span_placeholder_index = d.add_symbol(span_placeholder)
        d.event_placeholder_index = d.add_symbol(event_placeholder)

        d.nspecial = 999
        return d

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(f, zip(ex_keys + self.symbols, ex_vals + self.count))

    def delimiter(self):
        return self.span_delimiter_index

    def span_bos(self):
        return self.span_bos_index

    def span_eos(self):
        return self.span_eos_index

    def span_placeholder(self):
        return self.span_placeholder_index

    def event_placeholder(self):
        return self.event_placeholder_index
