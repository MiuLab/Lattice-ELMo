import random
import math
import time
import logging
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, Tuple, List, Union, Dict, Any, Callable

from torch.nn.modules import Dropout
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.elmo import _ElmoBiLm as ElmoBiLm
from allennlp.modules.elmo import Elmo, _ElmoCharacterEncoder
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.lstm_cell_with_projection import LstmCellWithProjection
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, sort_batch_by_length
from allennlp.nn.util import remove_sentence_boundaries
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.nn.util import get_dropout_mask

RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
logger = logging.getLogger(__name__)


class RNNModel(nn.Module):
    def __init__(self, config):
        super(RNNModel, self).__init__()
        self.config = config

        self.dim_embedding = config["dim_embedding"]
        self.dim_elmo_embedding = config.get("dim_elmo_embedding", 0)
        if self.dim_elmo_embedding > 0:
            self.dim_rnn_input = self.dim_elmo_embedding
        else:
            self.dim_rnn_input = self.dim_embedding
        # self.dim_rnn_input = self.dim_embedding + self.dim_elmo_embedding
        self.dim_hidden = config["dim_hidden"]
        self.vocab_size = config["vocab_size"]
        self.label_vocab_size = config["label_vocab_size"]
        self.n_layers = config["n_layers"]
        self.dropout_embedding = config["dropout_embedding"]
        self.dropout_hidden = config["dropout_hidden"]
        self.dropout_output = config["dropout_output"]
        self.bidirectional = config["bidirectional"]
        self.n_directions = 2 if self.bidirectional else 1
        self.pooling = config.get("pooling", "max")
        if self.pooling not in {"max", "mean", "last"}:
            raise ValueError(f"Pooling method '{self.pooling}' is not supported")

        self.dropout = LockedDropout()

    def get_pooled_output(self, output, positions):
        if self.pooling == "max":
            pooled_output = torch.stack([
                output[i, pos[-2]:pos[-1]].max(dim=0)[0]
                if pos[-2] != pos[-1] else torch.zeros(output.size(2)).to(output.device)
                for i, pos in enumerate(positions)
            ], dim=0)
        elif self.pooling == "mean":
            pooled_output = torch.stack([
                output[i, pos[-2]:pos[-1]].mean(dim=0)
                if pos[-2] != pos[-1] else torch.zeros(output.size(2)).to(output.device)
                for i, pos in enumerate(positions)
            ], dim=0)
        elif self.pooling == "last":
            if self.bidirectional:
                forward_output = torch.stack([
                    output[i, pos[-1]-1, :self.dim_hidden]
                    for i, pos in enumerate(positions)
                ], dim=0)
                backward_output = torch.stack([
                    output[i, 0, self.dim_hidden:]
                    for i, pos in enumerate(positions)
                ], dim=0)
                pooled_output = torch.cat(
                    (forward_output, backward_output), dim=-1)
            else:
                pooled_output = torch.stack([
                    output[i, pos[-1]-1]
                    for i, pos in enumerate(positions)
                ], dim=0)
        else:
            raise ValueError(f"Pooling method '{self.pooling}' is not supported")

        return pooled_output

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class SLURNN(RNNModel):
    def __init__(self, *args, **kwargs):
        super(SLURNN, self).__init__(*args, **kwargs)

        self.embedding = nn.Embedding(self.vocab_size, self.dim_embedding)
        """
        self.bilstm = nn.LSTM(self.dim_rnn_input,
            self.dim_hidden,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=self.bidirectional)
        """
        self.rnn = nn.ModuleList([
            nn.LSTM(
                self.dim_rnn_input if i == 0 else self.dim_hidden * self.n_directions,
                self.dim_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
            for i in range(self.n_layers)
        ])

        if self.bidirectional:
            self.rev_rnn = nn.ModuleList([
                nn.LSTM(
                    self.dim_rnn_input if i == 0 else self.dim_hidden * self.n_directions,
                    self.dim_hidden,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False
                )
                for i in range(self.n_layers)
            ])

        """
        for rnn, postfix in ((self.rnn[0], "l0"), (self.rnn[1], "l1"), (self.rev_rnn[0], "l0_reverse"), (self.rev_rnn[1], "l1_reverse")):
            self.copy_lstm(rnn, self.bilstm, "l0", postfix)
        """
        self.linear = nn.Linear(
            self.dim_hidden * self.n_directions,
            self.label_vocab_size)
        self.init_weights()

    def copy_lstm(self, lstm1, lstm2, lstm1_postfix="", lstm2_postfix=""):
        for prefix in ["weight_hh", "weight_ih", "bias_hh", "bias_ih"]:
            getattr(lstm1, f"{prefix}_{lstm1_postfix}").data.copy_(getattr(lstm2, f"{prefix}_{lstm2_postfix}").data)

    def forward(self, inputs, positions, elmo_emb=None):
        """
        args:
            inputs: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, label_vocab_size]
        """

        bs, sl = inputs.size()
        if elmo_emb is None:
            inputs = self.embedding(inputs)
            # if elmo_emb is not None:
            #    inputs = torch.cat([inputs, elmo_emb], dim=2)
        else:
            inputs = elmo_emb

        # inputs = self.dropout(inputs, self.dropout_embedding)
        # output, _ = self.bilstm(inputs)

        last_output = inputs
        for l, rnn in enumerate(self.rnn):
            output, _ = rnn(last_output)

            if self.bidirectional:
                rev_rnn = self.rev_rnn[l]
                rev_output, _ = rev_rnn(last_output.flip(dims=[1]))
                output = torch.cat((output, rev_output.flip(dims=[1])), dim=2)

            # if l != self.n_layers - 1:
            #    output = self.dropout(output, self.dropout_hidden)
            last_output = output

        # output = self.dropout(output, self.dropout_output)
        pooled_output = self.get_pooled_output(output, positions)
        logits = self.linear(pooled_output)
        return logits

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)


class LatticeRNN(nn.Module):
    def __init__(self, dim_input, dim_output, combine_method="weighted-sum"):
        super(LatticeRNN, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.combine_method = combine_method
        if self.combine_method not in ["weighted-sum", "sum", "best"]:
            raise ValueError(f"combine_method '{combine_method}' not supported")

        self.rnn = nn.LSTM(
            self.dim_input,
            self.dim_output,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

    def get_pooled_states(self, t, prevs, hs, cs):
        h, c = [], []
        for b, prev in enumerate(prevs):
            if len(prev[t]) == 1:
                idx, _ = prev[t][0]
                h.append(hs[idx][b])
                c.append(cs[idx][b])
                continue

            if self.combine_method in ["weighted-sum", "sum"]:
                tmp_hs, tmp_cs = [], []
                for idx, marginal in prev[t]:
                    if self.combine_method == "weighted-sum":
                        tmp_hs.append(marginal * hs[idx][b])
                        tmp_cs.append(marginal * cs[idx][b])
                    else:
                        tmp_hs.append(hs[idx][b])
                        tmp_cs.append(cs[idx][b])
                h.append(torch.stack(tmp_hs, dim=0).sum(dim=0))
                c.append(torch.stack(tmp_cs, dim=0).sum(dim=0))
            elif self.combine_method == "best":
                best_idx, _ = sorted(prev[t], key=lambda x:x[1], reverse=True)[0]
                h.append(hs[best_idx][b])
                c.append(cs[best_idx][b])

        h = torch.stack(h, dim=1)
        c = torch.stack(c, dim=1)

        return h, c

    def forward(self, inputs, prevs):
        """
        args:
            inputs: shape [batch_size, seq_length, dim_input]
            prevs: list of list of list of integers
                the indices of previous words at each step

        outputs:
            outputs: shape [batch_size, seq_length, dim_output]
        """
        bs, sl, _ = inputs.size()

        device = inputs.device
        inputs = inputs.split(1, dim=1)

        h = torch.zeros((1, bs, self.rnn.hidden_size)).to(device)
        c = torch.zeros((1, bs, self.rnn.hidden_size)).to(device)
        outputs = []
        hs = []
        cs = []
        for t in range(sl):
            if t != 0:
                h, c = self.get_pooled_states(t, prevs, hs, cs)
                h = h.to(device)
                c = c.to(device)
            output, (h, c) = self.rnn(inputs[t], (h, c))
            hs.append(h.cpu().squeeze(0).split(1, dim=0))
            cs.append(c.cpu().squeeze(0).split(1, dim=0))
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0).squeeze(2).transpose(0, 1)
        return outputs


class SLULatticeRNN(RNNModel):
    def __init__(self, *args, **kwargs):
        super(SLULatticeRNN, self).__init__(*args, **kwargs)

        self.combine_method = self.config.get("combine_method", "weighted-sum")
        self.embedding = nn.Embedding(self.vocab_size, self.dim_embedding)
        self.rnn = nn.ModuleList([
            LatticeRNN(
                self.dim_rnn_input if i == 0 else self.dim_hidden * self.n_directions,
                self.dim_hidden,
                combine_method=self.combine_method
            )
            for i in range(self.n_layers)
        ])

        if self.bidirectional:
            self.rev_rnn = nn.ModuleList([
                LatticeRNN(
                    self.dim_rnn_input if i == 0 else self.dim_hidden * self.n_directions,
                    self.dim_hidden,
                    combine_method=self.combine_method
                )
                for i in range(self.n_layers)
            ])

        self.linear = nn.Linear(
            self.dim_hidden * self.n_directions,
            self.label_vocab_size)
        self.init_weights()

    def forward(self, inputs, positions, prevs,
                rev_inputs=None, rev_prevs=None, elmo_emb=None):
        """
        args:
            inputs: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, label_vocab_size]
        """
        bs, sl = inputs.size()

        if elmo_emb is None:
            inputs = self.embedding(inputs)
            if rev_inputs is not None:
                rev_inputs = self.embedding(rev_inputs)
        else:
            inputs = elmo_emb
            rev_inputs = elmo_emb.flip(dims=[1])

        device = inputs.device
        # inputs = self.dropout(inputs, self.dropout_embedding)
        # rev_inputs = self.dropout(rev_inputs, self.dropout_embedding)
        last_output = inputs
        for l, rnn in enumerate(self.rnn):
            output = rnn(last_output, prevs)
            # if l != self.n_layers - 1:
            #    output = self.dropout(output, self.dropout_hidden)
            if self.bidirectional:
                if l == 0:
                    last_output_rev = rev_inputs
                else:
                    last_output_rev = last_output.flip(dims=[1])
                rev_rnn = self.rev_rnn[l]
                rev_output = rev_rnn(last_output_rev, rev_prevs)
                # if l != self.n_layers - 1:
                #     rev_output = self.dropout(rev_output, self.dropout_hidden)
                output = torch.cat((output, rev_output.flip(dims=[1])), dim=2)

            last_output = output

        # output = self.dropout(output, self.dropout_output)
        pooled_output = self.get_pooled_output(output, positions)
        logits = self.linear(pooled_output)
        return logits

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.requires_grad_(False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class ELMoLM(nn.Module):
    def __init__(self, option_file, weight_file, vocab_size):
        super().__init__()
        self.elmo = ElmoBiLm(option_file, weight_file, requires_grad=True)
        self.output_dim = self.elmo.get_output_dim()
        self.output_dim_half = self.output_dim // 2
        self.decoder = nn.Linear(self.output_dim_half, vocab_size)

    def forward(self, inputs):
        bilm_output = self.elmo(inputs)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        representations = []
        for representation in layer_activations:
            r, mask = remove_sentence_boundaries(representation, mask_with_bos_eos)
            representations.append(r)

        repr_forward, repr_backward = representations[-1].split(self.output_dim_half, dim=2)
        logits_forward = self.decoder(repr_forward)
        logits_backward = self.decoder(repr_backward)

        return logits_forward, logits_backward, representations, mask


class LatticeELMoLM(nn.Module):
    def __init__(self, option_file, weight_file, vocab_size,
                 combine_method="weighted-sum",
                 random_init=False):
        super().__init__()
        self.elmo = _LatticeElmoBiLm(option_file, weight_file,
                                     requires_grad=True,
                                     combine_method=combine_method,
                                     random_init=random_init)
        self.output_dim = self.elmo.get_output_dim()
        self.output_dim_half = self.output_dim // 2
        self.decoder = nn.Linear(self.output_dim_half, vocab_size)

    def forward(self, inputs, prevs, rev_prevs):
        bilm_output = self.elmo(inputs, prevs=prevs, rev_prevs=rev_prevs)
        representations = bilm_output['activations']

        """
        representations = []
        for representation in layer_activations:
            r, mask = remove_sentence_boundaries(representation, mask_with_bos_eos)
            representations.append(r)
        """
        repr_forward, repr_backward = representations[-1].split(self.output_dim_half, dim=2)
        logits_forward = self.decoder(repr_forward)
        logits_backward = self.decoder(repr_backward)

        return logits_forward, logits_backward, representations


class LatticeElmo(Elmo):
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        num_output_representations: int,
        requires_grad: bool = False,
        do_layer_norm: bool = False,
        dropout: float = 0.5,
        vocab_to_cache: List[str] = None,
        keep_sentence_boundaries: bool = True,
        scalar_mix_parameters: List[float] = None,
        module: torch.nn.Module = None,
        combine_method = "weighted-sum",
        random_init = False
    ) -> None:
        super(Elmo, self).__init__()

        logger.info("Initializing ELMo")
        self._elmo_lstm = _LatticeElmoBiLm(
            options_file,
            weight_file,
            requires_grad=requires_grad,
            vocab_to_cache=vocab_to_cache,
            combine_method=combine_method,
            random_init=random_init
        )
        self._has_cached_vocab = vocab_to_cache is not None
        self._keep_sentence_boundaries = keep_sentence_boundaries
        self._dropout = Dropout(p=dropout)
        self._scalar_mixes: Any = []
        for k in range(num_output_representations):
            scalar_mix = ScalarMix(
                self._elmo_lstm.num_layers,
                do_layer_norm=do_layer_norm,
                initial_scalar_parameters=scalar_mix_parameters,
                trainable=scalar_mix_parameters is None,
            )
            self.add_module("scalar_mix_{}".format(k), scalar_mix)
            self._scalar_mixes.append(scalar_mix)

    def forward(
        self, inputs: torch.Tensor, word_inputs: torch.Tensor = None,
        prevs=None, rev_prevs=None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        # reshape the input if needed
        original_shape = inputs.size()
        if len(original_shape) > 3:
            timesteps, num_characters = original_shape[-2:]
            reshaped_inputs = inputs.view(-1, timesteps, num_characters)
        else:
            reshaped_inputs = inputs

        reshaped_word_inputs = word_inputs

        # run the biLM
        bilm_output = self._elmo_lstm(
            reshaped_inputs, reshaped_word_inputs, prevs, rev_prevs)
        layer_activations = bilm_output["activations"]
        mask_with_bos_eos = bilm_output["mask"]

        # compute the elmo representations
        representations = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, "scalar_mix_{}".format(i))
            representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
            if self._keep_sentence_boundaries:
                processed_representation = representation_with_bos_eos
                processed_mask = mask_with_bos_eos
            else:
                representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(
                    representation_with_bos_eos, mask_with_bos_eos
                )
                processed_representation = representation_without_bos_eos
                processed_mask = mask_without_bos_eos
            representations.append(self._dropout(processed_representation))

        # reshape if necessary
        if word_inputs is not None and len(original_word_size) > 2:
            mask = processed_mask.view(original_word_size)
            elmo_representations = [
                representation.view(original_word_size + (-1,))
                for representation in representations
            ]
        elif len(original_shape) > 3:
            mask = processed_mask.view(original_shape[:-1])
            elmo_representations = [
                representation.view(original_shape[:-1] + (-1,))
                for representation in representations
            ]
        else:
            mask = processed_mask
            elmo_representations = representations

        return {"elmo_representations": elmo_representations, "mask": mask}


class _LatticeElmoBiLm(ElmoBiLm):
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        requires_grad: bool = False,
        vocab_to_cache: List[str] = None,
        combine_method = "weighted-sum",
        random_init = False
    ) -> None:
        super(ElmoBiLm, self).__init__()

        self._token_embedder = _ElmoCharacterEncoder(
            options_file, weight_file, requires_grad=requires_grad
        )

        self._requires_grad = requires_grad
        self._word_embedding = None
        self._bos_embedding: torch.Tensor = None
        self._eos_embedding: torch.Tensor = None

        with open(cached_path(options_file), "r") as fin:
            options = json.load(fin)

        if not options["lstm"].get("use_skip_connections"):
            raise ConfigurationError("We only support pretrained biLMs with residual connections")

        self._elmo_lstm = LatticeElmoLstm(
            input_size=options["lstm"]["projection_dim"],
            hidden_size=options["lstm"]["projection_dim"],
            cell_size=options["lstm"]["dim"],
            num_layers=options["lstm"]["n_layers"],
            memory_cell_clip_value=options["lstm"]["cell_clip"],
            state_projection_clip_value=options["lstm"]["proj_clip"],
            requires_grad=requires_grad,
            combine_method=combine_method
        )

        if not random_init:
            self._elmo_lstm.load_weights(weight_file)
        else:
            print("WARNING!!! ELMo weights will not be loaded!!!")
        # Number of representation layers including context independent layer
        self.num_layers = options["lstm"]["n_layers"] + 1

    def forward(
        self, inputs: torch.Tensor, word_inputs: torch.Tensor = None,
        prevs=None, rev_prevs=None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        # discard the original <s> </s> tokens in lattices
        padding = inputs.new_zeros(inputs.size(-1))
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()
        sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
        for i, l in enumerate(sequence_lengths):
            inputs[i, l-1] = padding
        inputs = inputs[:, 1:-1]

        token_embedding = self._token_embedder(inputs)
        mask = token_embedding["mask"]
        type_representation = token_embedding["token_embedding"]
        lstm_outputs = self._elmo_lstm(type_representation, mask, prevs, rev_prevs)

        output_tensors = [
            torch.cat([type_representation, type_representation], dim=-1)
            * mask.float().unsqueeze(-1)
        ]
        for layer_activations in torch.chunk(lstm_outputs, lstm_outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))

        return {"activations": output_tensors, "mask": mask}


class LatticeElmoLstm(ElmoLstm):
    def __init__(self,
                input_size: int,
                hidden_size: int,
                cell_size: int,
                num_layers: int,
                requires_grad: bool = False,
                recurrent_dropout_probability: float = 0.0,
                memory_cell_clip_value: Optional[float] = None,
                state_projection_clip_value: Optional[float] = None,
                combine_method="weighted-sum"):
        super(ElmoLstm, self).__init__(stateful=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_size = cell_size
        self.requires_grad = requires_grad

        forward_layers = []
        backward_layers = []

        lstm_input_size = input_size
        go_forward = True
        for layer_index in range(num_layers):
            forward_layer = LatticeLstmCellWithProjection(
                lstm_input_size,
                hidden_size,
                cell_size,
                go_forward,
                recurrent_dropout_probability,
                memory_cell_clip_value,
                state_projection_clip_value,
                combine_method=combine_method
            )
            backward_layer = LatticeLstmCellWithProjection(
                lstm_input_size,
                hidden_size,
                cell_size,
                not go_forward,
                recurrent_dropout_probability,
                memory_cell_clip_value,
                state_projection_clip_value,
                combine_method=combine_method
            )
            lstm_input_size = hidden_size

            self.add_module("forward_layer_{}".format(layer_index), forward_layer)
            self.add_module("backward_layer_{}".format(layer_index), backward_layer)
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)
        self.forward_layers = forward_layers
        self.backward_layers = backward_layers

    def forward(self,
                inputs: torch.Tensor,
                mask: torch.LongTensor,
                prevs=None,
                rev_prevs=None) -> torch.Tensor:
        batch_size, total_sequence_length = mask.size()
        stacked_sequence_output, final_states, restoration_indices = self.sort_and_run_forward(
            self._lstm_forward, inputs, mask, prevs=prevs, rev_prevs=rev_prevs
        )

        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
        # Add back invalid rows which were removed in the call to sort_and_run_forward.
        if num_valid < batch_size:
            zeros = stacked_sequence_output.new_zeros(
                num_layers, batch_size - num_valid, returned_timesteps, encoder_dim
            )
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

            # The states also need to have invalid rows added back.
            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.new_zeros(num_layers, batch_size - num_valid, state_dim)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2StackEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - returned_timesteps
        if sequence_length_difference > 0:
            zeros = stacked_sequence_output.new_zeros(
                num_layers,
                batch_size,
                sequence_length_difference,
                stacked_sequence_output[0].size(-1),
            )
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)

        self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        # Has shape (num_layers, batch_size, sequence_length, hidden_size)
        return stacked_sequence_output.index_select(1, restoration_indices)

    def _lstm_forward(
        self,
        inputs: PackedSequence,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        prevs=None, rev_prevs=None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM, with shape (num_layers, batch_size, 2 * hidden_size) and
            (num_layers, batch_size, 2 * cell_size) respectively.
        Returns
        -------
        output_sequence : ``torch.FloatTensor``
            The encoded sequence of shape (num_layers, batch_size, sequence_length, hidden_size)
        final_states: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
            The per-layer final (state, memory) states of the LSTM, with shape
            (num_layers, batch_size, 2 * hidden_size) and  (num_layers, batch_size, 2 * cell_size)
            respectively. The last dimension is duplicated because it contains the state/memory
            for both the forward and backward layers.
        """
        if initial_state is None:
            hidden_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(
                self.forward_layers
            )
        elif initial_state[0].size()[0] != len(self.forward_layers):
            raise ConfigurationError(
                "Initial states were passed to forward() but the number of "
                "initial states does not match the number of layers."
            )
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

        inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        forward_output_sequence = inputs
        backward_output_sequence = inputs

        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(hidden_states):
            forward_layer = getattr(self, "forward_layer_{}".format(layer_index))
            backward_layer = getattr(self, "backward_layer_{}".format(layer_index))

            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
                forward_state = (forward_hidden_state, forward_memory_state)
                backward_state = (backward_hidden_state, backward_memory_state)
            else:
                forward_state = None
                backward_state = None

            forward_output_sequence, forward_state = forward_layer(
                forward_output_sequence, batch_lengths, forward_state, prevs
            )
            backward_output_sequence, backward_state = backward_layer(
                backward_output_sequence, batch_lengths, backward_state, rev_prevs
            )
            # Skip connections, just adding the input to the output.
            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache

            sequence_outputs.append(
                torch.cat([forward_output_sequence, backward_output_sequence], -1)
            )
            # Append the state tuples in a list, so that we can return
            # the final states for all the layers.
            final_states.append(
                (
                    torch.cat([forward_state[0], backward_state[0]], -1),
                    torch.cat([forward_state[1], backward_state[1]], -1),
                )
            )

        stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
        # Stack the hidden state and memory for each layer into 2 tensors of shape
        # (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
        # respectively.
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple: Tuple[torch.FloatTensor, torch.FloatTensor] = (
            torch.cat(final_hidden_states, 0),
            torch.cat(final_memory_states, 0),
        )
        return stacked_sequence_outputs, final_state_tuple

    def sort_and_run_forward(
        self,
        module: Callable[
            [PackedSequence, Optional[RnnState]],
            Tuple[Union[PackedSequence, torch.Tensor], RnnState],
        ],
        inputs: torch.Tensor,
        mask: torch.Tensor,
        hidden_state: Optional[RnnState] = None,
        prevs=None,
        rev_prevs=None
    ):
        # In some circumstances you may have sequences of zero length. ``pack_padded_sequence``
        # requires all sequence lengths to be > 0, so remove sequences of zero length before
        # calling self._module, then fill with zeros.

        # First count how many sequences are empty.
        batch_size = mask.size(0)
        num_valid = torch.sum(mask[:, 0]).int().item()

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, sorting_indices = sort_batch_by_length(
            inputs, sequence_lengths
        )

        prevs = [prevs[i] for i in sorting_indices][:num_valid]
        rev_prevs = [rev_prevs[i] for i in sorting_indices][:num_valid]

        # Now create a PackedSequence with only the non-empty, sorted sequences.
        packed_sequence_input = pack_padded_sequence(
            sorted_inputs[:num_valid, :, :],
            sorted_sequence_lengths[:num_valid].data.tolist(),
            batch_first=True,
        )
        # Prepare the initial states.
        if not self.stateful:
            if hidden_state is None:
                initial_states: Any = hidden_state
            elif isinstance(hidden_state, tuple):
                initial_states = [
                    state.index_select(1, sorting_indices)[:, :num_valid, :].contiguous()
                    for state in hidden_state
                ]
            else:
                initial_states = hidden_state.index_select(1, sorting_indices)[
                    :, :num_valid, :
                ].contiguous()
        else:
            initial_states = self._get_initial_states(batch_size, num_valid, sorting_indices)

        # Actually call the module on the sorted PackedSequence.
        module_output, final_states = module(packed_sequence_input, initial_states, prevs, rev_prevs)

        return module_output, final_states, restoration_indices


class LatticeLstmCellWithProjection(LstmCellWithProjection):
    def __init__(self, *args, **kwargs):
        combine_method = kwargs.pop("combine_method")
        super().__init__(*args, **kwargs)
        self.combine_method = combine_method

    def get_pooled_states(self, t, prevs, hs, cs, current_length_index):
        h, c = [], []
        for b, prev in enumerate(prevs):
            if b == current_length_index + 1:
                break

            if len(prev[t]) == 1:
                idx, _ = prev[t][0]
                h.append(hs[idx][b])
                c.append(cs[idx][b])
                continue

            if self.combine_method in ["weighted-sum", "sum"]:
                tmp_hs, tmp_cs = [], []
                for idx, marginal in prev[t]:
                    if self.combine_method == "weighted-sum":
                        tmp_hs.append(marginal * hs[idx][b])
                        tmp_cs.append(marginal * cs[idx][b])
                    else:
                        tmp_hs.append(hs[idx][b])
                        tmp_cs.append(cs[idx][b])
                h.append(torch.stack(tmp_hs, dim=0).sum(dim=0))
                c.append(torch.stack(tmp_cs, dim=0).sum(dim=0))
            elif self.combine_method == "best":
                best_idx, _ = sorted(prev[t], key=lambda x:x[1], reverse=True)[0]
                h.append(hs[best_idx][b])
                c.append(cs[best_idx][b])

        h = torch.stack(h, dim=0)
        c = torch.stack(c, dim=0)

        return h, c

    def forward(
        self,
        inputs: torch.FloatTensor,
        batch_lengths: List[int],
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        prevs = None
    ):
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.
        batch_lengths : ``List[int]``, required.
            A list of length batch_size containing the lengths of the sequences in batch.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).
        Returns
        -------
        output_accumulator : ``torch.FloatTensor``
            The outputs of the LSTM for each timestep. A tensor of shape
            (batch_size, max_timesteps, hidden_size) where for a given batch
            element, all outputs past the sequence length for that batch are
            zero tensors.
        final_state : ``Tuple[``torch.FloatTensor, torch.FloatTensor]``
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).
        """
        batch_size = inputs.size()[0]
        total_timesteps = inputs.size()[1]

        output_accumulator = inputs.new_zeros(batch_size, total_timesteps, self.hidden_size)

        if initial_state is None:
            full_batch_previous_memory = inputs.new_zeros(batch_size, self.cell_size)
            full_batch_previous_state = inputs.new_zeros(batch_size, self.hidden_size)
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)

        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(
                self.recurrent_dropout_probability, full_batch_previous_state
            )
        else:
            dropout_mask = None

        hs, cs = [], []

        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            else:
                while (
                    current_length_index < (len(batch_lengths) - 1)
                    and batch_lengths[current_length_index + 1] > index
                ):
                    current_length_index += 1

            if timestep != 0:
                previous_memory, previous_state = self.get_pooled_states(
                    timestep, prevs, hs, cs, current_length_index)
            else:
                previous_memory = full_batch_previous_memory[0 : current_length_index + 1].clone()
                previous_state = full_batch_previous_state[0 : current_length_index + 1].clone()

            timestep_input = inputs[0 : current_length_index + 1, index]

            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)

            input_gate = torch.sigmoid(
                projected_input[:, (0 * self.cell_size) : (1 * self.cell_size)]
                + projected_state[:, (0 * self.cell_size) : (1 * self.cell_size)]
            )
            forget_gate = torch.sigmoid(
                projected_input[:, (1 * self.cell_size) : (2 * self.cell_size)]
                + projected_state[:, (1 * self.cell_size) : (2 * self.cell_size)]
            )
            memory_init = torch.tanh(
                projected_input[:, (2 * self.cell_size) : (3 * self.cell_size)]
                + projected_state[:, (2 * self.cell_size) : (3 * self.cell_size)]
            )
            output_gate = torch.sigmoid(
                projected_input[:, (3 * self.cell_size) : (4 * self.cell_size)]
                + projected_state[:, (3 * self.cell_size) : (4 * self.cell_size)]
            )
            memory = input_gate * memory_init + forget_gate * previous_memory

            if self.memory_cell_clip_value:
                memory = torch.clamp(
                    memory, -self.memory_cell_clip_value, self.memory_cell_clip_value
                )

            # shape (current_length_index, cell_size)
            pre_projection_timestep_output = output_gate * torch.tanh(memory)

            # shape (current_length_index, hidden_size)
            timestep_output = self.state_projection(pre_projection_timestep_output)
            if self.state_projection_clip_value:
                timestep_output = torch.clamp(
                    timestep_output,
                    -self.state_projection_clip_value,
                    self.state_projection_clip_value,
                )
            if dropout_mask is not None:
                timestep_output = timestep_output * dropout_mask[0 : current_length_index + 1]

            full_batch_previous_memory = full_batch_previous_memory.clone()
            full_batch_previous_state = full_batch_previous_state.clone()
            full_batch_previous_memory[0 : current_length_index + 1] = memory
            full_batch_previous_state[0 : current_length_index + 1] = timestep_output
            hs.append(full_batch_previous_memory)
            cs.append(full_batch_previous_state)
            output_accumulator[0 : current_length_index + 1, index] = timestep_output

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, ...). As this
        # LSTM cell cannot be stacked, the first dimension here is just 1.
        final_state = (
            full_batch_previous_state.unsqueeze(0),
            full_batch_previous_memory.unsqueeze(0),
        )

        return output_accumulator, final_state
