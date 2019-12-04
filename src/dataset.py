import pickle
import os
import json
import csv
import re

import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm

from vocabulary import Vocab, PAD
from model_utils import pad_sequences, pad_matrices, index_to_nhot
from utils import print_time_info
from lattice_utils import LatticeReader


class SLUDataset(Dataset):
    label_idx = 3

    def __init__(self, filename, vocab_file=None,
                 vocab_dump=None, label_vocab_dump=None,
                 n_prev_turns=0):
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            self.data = [row for row in reader]

        if "id" in self.data[0]:
            self.id2idx = {row["id"]: i for i, row in enumerate(self.data)}

        self.n_prev_turns = n_prev_turns
        if vocab_dump is None:
            self.vocab = Vocab(vocab_file)
        else:
            with open(vocab_dump, 'rb') as fp:
                self.vocab = pickle.load(fp)
        if label_vocab_dump is None:
            labels = [row["label"] for row in self.data]
            self.label_vocab = LabelVocab(labels)
        else:
            with open(label_vocab_dump, 'rb') as fp:
                self.label_vocab = pickle.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _process_text(self, text):
        for punct in [',', '.', '?', '!']:
            if text.endswith(f" {punct}"):
                text = text[:-2]
        text = re.sub(" ([a-z])\. ", " \\1 ", text)
        return text

    def collate_fn(self, batch):
        inputs, words, positions, labels = [], [], [], []
        for utt in batch:
            prev_utts = []
            poss = [0]
            prev_id = utt.get("previous", "")
            while len(prev_utts) < self.n_prev_turns and prev_id != "":
                if prev_id not in self.id2idx:
                    break
                prev_row = self.data[self.id2idx[prev_id]]
                prev_id = prev_row["previous"]
                text = self._process_text(prev_row["text"])
                prev_utts = [text] + prev_utts

            text = self._process_text(utt["text"])
            for prev_utt in prev_utts:
                poss.append(poss[-1] + len(prev_utt.split()))
            poss.append(poss[-1] + len(text.split()))
            while len(poss) - 1 < self.n_prev_turns + 1:
                poss = [0] + poss
            label = utt["label"]
            text = " ".join(prev_utts + [text])
            word_ids = [self.vocab.w2i(word) for word in text.split()]
            words.append(text.split())
            inputs.append(word_ids)
            positions.append(poss)
            labels.append(self.label_vocab.l2i(label))

        max_length = max(map(len, inputs))
        inputs = pad_sequences(inputs, max_length)
        labels = np.array(labels)
        return inputs, words, positions, labels


class PairDataset(Dataset):
    label_idx = 3

    def __init__(self, filename, vocab_file=None,
                 vocab_dump=None, label_vocab_dump=None):
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            self.data = [row for row in reader]

        if vocab_dump is None:
            self.vocab = Vocab(vocab_file)
        else:
            with open(vocab_dump, 'rb') as fp:
                self.vocab = pickle.load(fp)
        if label_vocab_dump is None:
            labels = [row["label"] for row in self.data]
            self.label_vocab = LabelVocab(labels)
        else:
            with open(label_vocab_dump, 'rb') as fp:
                self.label_vocab = pickle.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _process_text(self, text):
        for punct in [',', '.', '?', '!']:
            if text.endswith(f" {punct}"):
                text = text[:-2]
        text = re.sub(" ([a-z])\. ", " \\1 ", text)
        return text

    def collate_fn(self, batch):
        inputs, words, positions, labels = [], [], [], []
        for utt in batch:
            text = self._process_text(utt["text"] + " " + utt["text2"])
            text = " ".join(text)
            label = utt["label"]
            word_ids = [self.vocab.w2i(word) for word in text.split()]
            words.append(text.split())
            inputs.append(word_ids)
            positions.append([0, len(word_ids)])
            labels.append(self.label_vocab.l2i(label))

        max_length = max(map(len, inputs))
        inputs = pad_sequences(inputs, max_length)
        labels = np.array(labels)
        return inputs, words, positions, labels


class SLULatticeDataset(Dataset):
    label_idx = 6

    def __init__(self, filename, vocab_file=None,
                 vocab_dump=None, label_vocab_dump=None,
                 n_prev_turns=0, text_input=False):
        self.text_input = text_input
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            self.data = [row for row in reader]
            lattice_reader = LatticeReader(text_input=text_input)
            for i, row in enumerate(tqdm(self.data)):
                row["lattice"] = lattice_reader.read_sent(row["text"], i)
                row["rev_lattice"] = row["lattice"].reversed()

        self.id2idx = {row["id"]: i for i, row in enumerate(self.data)}
        self.n_prev_turns = n_prev_turns
        if vocab_dump is None:
            self.vocab = Vocab(vocab_file)
        else:
            with open(vocab_dump, 'rb') as fp:
                self.vocab = pickle.load(fp)
        if label_vocab_dump is None:
            labels = [row["label"] for row in self.data]
            self.label_vocab = LabelVocab(labels)
        else:
            with open(label_vocab_dump, 'rb') as fp:
                self.label_vocab = pickle.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _process_text(self, text):
        for punct in [',', '.', '?', '!']:
            if text.endswith(f" {punct}"):
                text = text[:-2]
        text = re.sub(" ([a-z])\. ", " \\1 ", text)
        return text

    def _get_prev_nodes(self, lattice):
        prevs = []
        for node in lattice.nodes:
            prevs.append([
                (n, np.exp(lattice.nodes[n].marginal_log_prob))
                for n in node.nodes_prev
            ])
        return prevs

    def _pad_prevs(self, prevs, max_length, pad_type="post"):
        def shift_index(prev):
            shift = max_length - len(prev)
            return [
                [(idx+shift, prob) for idx, prob in p] if len(p) > 0 else [(-1, 1.0)]
                for p in prev
            ]

        if pad_type == "post":
            prevs = [
                prev + [[(-1, 1.0)] for _ in range(max_length-len(prev))]
                for prev in prevs
            ]
        elif pad_type == "pre":
            prevs = [
                [[(-1, 1.0)] for _ in range(max_length-len(prev))] + shift_index(prev)
                for prev in prevs
            ]
        return prevs

    def collate_fn(self, batch):
        inputs, words, positions, prevs = [], [], [], []
        rev_inputs, rev_prevs, labels = [], [], []
        for utt in batch:
            text = " ".join(utt["lattice"].str_tokens())
            word_ids = [self.vocab.w2i(word) for word in text.split()]
            prev = self._get_prev_nodes(utt["lattice"])
            rev_text = " ".join(utt["rev_lattice"].str_tokens())
            rev_word_ids = [self.vocab.w2i(word) for word in rev_text.split()]
            rev_prev = self._get_prev_nodes(utt["rev_lattice"])

            label = utt["label"]
            words.append(text.split())
            inputs.append(word_ids)
            positions.append([0, len(word_ids)])
            prevs.append(prev)
            rev_inputs.append(rev_word_ids)
            rev_prevs.append(rev_prev)
            labels.append(self.label_vocab.l2i(label))

        max_length = max(map(len, inputs))
        inputs = pad_sequences(inputs, max_length)
        rev_inputs = pad_sequences(rev_inputs, max_length, "pre")
        prevs = self._pad_prevs(prevs, max_length)
        rev_prevs = self._pad_prevs(rev_prevs, max_length, "pre")
        labels = np.array(labels)
        return inputs, words, positions, prevs, rev_inputs, rev_prevs, labels


class LabelVocab:
    def __init__(self, labels):
        self.build_vocab(labels)

    def build_vocab(self, labels):
        unique_labels = set(labels)
        sorted_labels = sorted(list(unique_labels))
        self.vocab = sorted_labels
        self.rev_vocab = dict()
        for i, label in enumerate(sorted_labels):
            self.rev_vocab[label] = i

    def l2i(self, label):
        try:
            return self.rev_vocab[label]
        except:
            raise KeyError(label)

    def i2l(self, index):
        return self.vocab[index]


class ConfusionDataset(Dataset):
    def __init__(self, filename, vocab_file=None, vocab_dump=None,
                 stop_word_file=None):
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            data = [row for row in reader]

        self.stop_words = set()
        if stop_word_file is not None:
            for line in open(stop_word_file):
                self.stop_words.add(line.strip())

        datas = []
        count, total = 0, 0
        for row in data:
            ref = row["transcription"]
            hyp = row["hypothesis"]
            score = float(row["score"])
            confs = row["confusion"].split()
            confs = [
                (confs[i*3], confs[i*3+1])
                for i in range(len(confs)//3+1)
            ]
            conf_ids = []
            ref_id = hyp_id = 0
            for ref_w, hyp_w in confs:
                ref_eps = (ref_w == "<eps>")
                hyp_eps = (hyp_w == "<eps>")
                if not ref_eps and not hyp_eps and ref_w != hyp_w:
                    total += 1
                    if ref_w not in self.stop_words and hyp_w not in self.stop_words:
                        conf_ids.append((ref_id, hyp_id))
                    else:
                        count += 1

                if not ref_eps:
                    ref_id += 1
                if not hyp_eps:
                    hyp_id += 1
            datas.append((ref, hyp, conf_ids, score))
        print(count, total)
        self.data = datas

        if vocab_file is not None:
            self.vocab = Vocab(vocab_file)
        elif vocab_dump is not None:
            with open(vocab_dump, 'rb') as fp:
                self.vocab = pickle.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, batch):
        transposed = zip(*batch)
        return tuple(transposed)

    def collate_fn_atis(self, batch):
        refs, ref_out_for, ref_out_rev = [], [], []
        hyps, hyp_out_for, hyp_out_rev = [], [], []
        confs, scores = [], []
        for ref, hyp, conf_ids, score in batch:
            ref = ref.strip().split()
            hyp = hyp.strip().split()
            confs.append(conf_ids)
            scores.append(score)
            refs.append(ref)
            ref_word_ids = [self.vocab.w2i(w) for w in ref]
            ref_out_for.append(ref_word_ids[1:] + [PAD])
            ref_out_rev.append([PAD] + ref_word_ids[:-1])
            hyps.append(hyp)
            hyp_word_ids = [self.vocab.w2i(w) for w in hyp]
            hyp_out_for.append(hyp_word_ids[1:] + [PAD])
            hyp_out_rev.append([PAD] + hyp_word_ids[:-1])

        inputs = refs + hyps
        outputs_for = ref_out_for + hyp_out_for
        outputs_rev = ref_out_rev + hyp_out_rev
        max_length = max([len(sent) for sent in outputs_for])

        outputs_for = pad_sequences(outputs_for, max_length, 'post')
        outputs_rev = pad_sequences(outputs_rev, max_length, 'post')

        return inputs, outputs_for, outputs_rev, confs, scores


class LMDataset(Dataset):
    def __init__(self, text_path, vocab_file=None, vocab_dump=None):
        self.data = []

        print_time_info("Reading text from {}".format(text_path))

        with open(text_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                words = row["text"].split()
                if "id" in row:
                    self.data.append((row["id"], words))
                else:
                    self.data.append((i, words))
        # for line in tqdm(open(text_path)):
        #     uid, *words = line.strip().split()
        #     self.data.append((uid, words))

        if vocab_dump is None:
            self.vocab = Vocab(vocab_file)
        else:
            with open(vocab_dump, 'rb') as fp:
                self.vocab = pickle.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        uid, sentence = self.data[index]
        word_ids = [self.vocab.w2i(word) for word in sentence]
        return uid, sentence, word_ids

    def collate_fn(self, batch):
        uids, inputs, outputs, outputs_rev = [], [], [], []
        for uid, words, word_ids in batch:
            uids.append(uid)
            inputs.append(words)
            outputs.append(word_ids[1:] + [PAD])
            outputs_rev.append([PAD] + word_ids[:-1])

        max_length = max([len(sent) for sent in outputs])
        # (batch_size, seq_length)
        outputs = pad_sequences(outputs, max_length, 'post')
        outputs_rev = pad_sequences(outputs_rev, max_length, 'post')

        return inputs, outputs, outputs_rev, uids


class LMLatticeDataset(Dataset):
    def __init__(self, filename, vocab_file=None,
                 vocab_dump=None, text_input=False):
        self.text_input = text_input
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            self.data = [row for row in reader]
            lattice_reader = LatticeReader(text_input=text_input)
            for i, row in enumerate(tqdm(self.data)):
                row["lattice"] = lattice_reader.read_sent(row["text"], i)
                row["rev_lattice"] = row["lattice"].reversed()

        if vocab_dump is None:
            self.vocab = Vocab(vocab_file)
        else:
            with open(vocab_dump, 'rb') as fp:
                self.vocab = pickle.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _process_text(self, text):
        for punct in [',', '.', '?', '!']:
            if text.endswith(f" {punct}"):
                text = text[:-2]
        text = re.sub(" ([a-z])\. ", " \\1 ", text)
        return text

    def _get_prev_nodes(self, lattice):
        prevs = []
        for node in lattice.nodes:
            if len(node.nodes_prev) == 0:
                prev_prob_sum = 1.0
            else:
                prev_prob_sum = sum([np.exp(lattice.nodes[n].marginal_log_prob) for n in node.nodes_prev])
            prevs.append([
                (n, np.exp(lattice.nodes[n].marginal_log_prob) / prev_prob_sum)
                for n in node.nodes_prev
            ])
        return prevs

    def _get_lm_labels(self, lattice):
        probs = []
        mask = []
        tokens = lattice.str_tokens()
        for node in lattice.nodes:
            if len(node.nodes_next) == 0:
                next_prob_sum = 1.0
            else:
                next_prob_sum = sum([np.exp(lattice.nodes[n].marginal_log_prob) for n in node.nodes_next])
            prob = np.zeros(len(self.vocab.vocab))
            for n in node.nodes_next:
                wid = self.vocab.w2i(tokens[n])
                prob[wid] = np.exp(lattice.nodes[n].marginal_log_prob) / next_prob_sum
            probs.append(prob)
            mask.append(1 if len(node.nodes_next) >= 1 else 0)
        return np.array(probs), np.array(mask)

    def _pad_prevs(self, prevs, max_length, pad_type="post"):
        def shift_index(prev):
            shift = max_length - len(prev)
            return [
                [(idx+shift, prob) for idx, prob in p] if len(p) > 0 else [(-1, 1.0)]
                for p in prev
            ]

        if pad_type == "post":
            prevs = [
                prev + [[(-1, 1.0)] for _ in range(max_length-len(prev))]
                for prev in prevs
            ]
        elif pad_type == "pre":
            prevs = [
                [[(-1, 1.0)] for _ in range(max_length-len(prev))] + shift_index(prev)
                for prev in prevs
            ]
        return prevs

    def collate_fn(self, batch):
        inputs, words, positions, prevs = [], [], [], []
        rev_inputs, rev_prevs, lm_labels, rev_lm_labels = [], [], [], []
        lm_masks, rev_lm_masks = [], []
        for utt in batch:
            text = " ".join(utt["lattice"].str_tokens())
            word_ids = [self.vocab.w2i(word) for word in text.split()]
            prev = self._get_prev_nodes(utt["lattice"])
            lm_label, lm_mask = self._get_lm_labels(utt["lattice"])
            rev_text = " ".join(utt["rev_lattice"].str_tokens())
            rev_word_ids = [self.vocab.w2i(word) for word in rev_text.split()]
            rev_prev = self._get_prev_nodes(utt["rev_lattice"])
            rev_lm_label, rev_lm_mask = self._get_lm_labels(utt["rev_lattice"])

            words.append(text.split())
            positions.append([0, len(word_ids)])
            prevs.append(prev)
            lm_labels.append(lm_label)
            lm_masks.append(lm_mask)
            rev_prevs.append(rev_prev)
            rev_lm_labels.append(rev_lm_label)
            rev_lm_masks.append(rev_lm_mask)

        max_length = max(map(len, lm_labels))
        prevs = self._pad_prevs(prevs, max_length)
        rev_prevs = self._pad_prevs(rev_prevs, max_length, "pre")
        lm_labels = pad_matrices(lm_labels, max_length)
        rev_lm_labels = pad_matrices(rev_lm_labels, max_length, "pre")
        lm_masks = pad_matrices(lm_masks, max_length)
        rev_lm_masks = pad_matrices(rev_lm_masks, max_length, "pre")
        return words, positions, prevs, rev_prevs, lm_labels, rev_lm_labels, lm_masks, rev_lm_masks


if __name__ == "__main__":
    dataset = ConfusionDataset('../data/csv/dev-asr-conf.csv')
    print(len(dataset))
    for i in range(2000, 2010):
        print(dataset[i])
