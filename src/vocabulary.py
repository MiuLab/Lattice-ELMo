import json

from tqdm import tqdm

from utils import print_time_info


BOS = 0
EOS = 1
PAD = 2
BOS_SYMBOL = '<BOS>'
EOS_SYMBOL = '<EOS>'
PAD_SYMBOL = '<PAD>'

class Vocab:
    def __init__(self, vocab_path):
        print_time_info("Reading vocabulary from {}".format(vocab_path))
        self.read_vocab(vocab_path)

    def read_vocab(self, vocab_path):
        self.vocab = dict()
        self.rev_vocab = dict()

        self.add_word(BOS_SYMBOL)
        self.add_word(EOS_SYMBOL)
        self.add_word(PAD_SYMBOL)
        self.bos_symbol = BOS_SYMBOL
        self.eos_symbol = EOS_SYMBOL
        self.pad_symbol = PAD_SYMBOL

        vocabs = set()
        for wid, line in tqdm(enumerate(open(vocab_path))):
            word = line.strip()
            vocabs.add(word)

        if '<unk>' in vocabs:
            self.unk_symbol = '<unk>'
        elif '<UNK>' in vocabs:
            self.unk_symbol = '<UNK>'
        else:
            self.unk_symbol = '<unk>'
        self.add_word(self.unk_symbol)

        for word in sorted(vocabs):
            self.add_word(word)

    def w2i(self, word):
        if word not in self.vocab:
            word = self.unk_symbol
        return self.vocab[word]

    def i2w(self, index):
        return self.rev_vocab[index]

    def add_word(self, word):
        if word in self.vocab:
            return
        wid = len(self.vocab)
        self.vocab[word] = wid
        self.rev_vocab[wid] = word

    def __len__(self):
        return len(self.vocab)


class Ontology:
    def __init__(self, ontology_file):
        with open(ontology_file, 'r') as f:
            ontology = json.load(f)
        self.requestable = ontology["requestable"]
        self.method = ontology["method"]
        self.informable = ontology["informable"]
        self.slot_list = list(sorted(self.informable.keys()))
        self.acts = ontology["acts"]
        self.svs = ontology["slot_value_pairs"]
        self.asvs = ontology["act_slot_value_triples"]

        self.requestable_rev = self._build_lookup_table(self.requestable)
        self.method_rev = self._build_lookup_table(self.method)
        self.slot_rev = self._build_lookup_table(self.slot_list)
        self.acts_rev = self._build_lookup_table(self.acts)
        self.svs_rev = self._build_lookup_table(self.svs)
        self.asvs_rev = self._build_lookup_table(self.asvs)
        self.informable_rev = {
            i: self._build_lookup_table(self.informable[slot])
            for i, slot in enumerate(self.slot_list)
        }
        self.informable_rev.update({
            slot: self._build_lookup_table(self.informable[slot])
            for slot in self.slot_list
        })

    def _build_lookup_table(self, list_of_names):
        rev = dict()
        for i, name in enumerate(list_of_names):
            if isinstance(name, list):
                name = tuple(name)
            rev[name] = i
        return rev

    def r2i(self, slot):
        return self.requestable_rev[slot]

    def i2r(self, index):
        return self.requestable[index]

    def m2i(self, method):
        return self.method_rev[method]

    def i2m(self, index):
        return self.method[index]

    def inf2i(self, slot, value):
        return self.slot_rev[slot], self.informable_rev[slot][value]

    def i2inf(self, index_slot, index_value):
        slot = self.slot_list[index_slot]
        value = self.informable[slot][index_value]
        return slot, value

    def sv2i(self, slot, value):
        return self.svs_rev[(slot, value)]

    def i2sv(self, index):
        return self.svs[index]

    def asv2i(self, act, slot=None, value=None):
        if slot is None or value is None:
            return self.asvs_rev[(act,)]
        return self.asvs_rev[(act, slot, value)]

    def i2asv(self, index):
        return self.asvs[index]

    def a2i(self, act):
        return self.acts_rev[act]

    def i2a(self, index):
        return self.acts[index]
