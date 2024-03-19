import os, platform, json, time, pickle, sys, argparse, random, datetime
import torch
from math import log
sys.path.append('./')
#from data_loader import Dataset_sentence_test, collate_func
#from model import LSTMEncoder, LSTMDecoder, Embeds
#from utils import Normlize_tx, Channel, smaple_n_times
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from ctypes import *
import ctypes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import namedtuple
from torch.utils.data import Dataset

from utils import smaple_n_times
from PlainTransformer import make_encoder, make_decoder, PositionalEncoding
import numpy as np

_snr = -25
_iscomplex = False # True 
channel_dim = 2

# Let's assume an SNR value (in dB) for demonstration
is_complex = True  # Considering the symbols as complex for this example

def get_model(name):
    return globals()[name]

class Decoder_Meta():
    
    # Note, `self` methods/attributes should be defined in child classes.
    def forward_rl(self, input_features, sample_max=None, multiple_sample=5, x_mask=None):

        if x_mask is not None:  # i.e., backbone is Transformer
            max_seq_len = input_features.shape[-1] // self.channel_dim
            input_features = smaple_n_times(multiple_sample, input_features.view(input_features.shape[0], max_seq_len, -1))
            input_features = self.from_channel_emb(input_features)
            x_mask = smaple_n_times(multiple_sample, x_mask)
        else: # LSTM
            def smaple_n_times(n, x):
                    if n>1:
                        x = x.unsqueeze(1) # Bx1x...
                        x = x.expand(-1, n, *([-1]*len(x.shape[2:])))
                        x = x.reshape(x.shape[0]*n, *x.shape[2:])
                    return x
            max_seq_len = 21  # we set the max sentence length to 20, plus an EOS token. You can adjust this value.
            input_features = smaple_n_times(multiple_sample, input_features)

        batch_size = input_features.size(0)
        state = self.init_hidden(input_features)

        seq = input_features.new_zeros((batch_size, max_seq_len), dtype=torch.long)
        seq_logprobs = input_features.new_zeros((batch_size, max_seq_len))
        seq_masks = input_features.new_zeros((batch_size, max_seq_len))
        it = input_features.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id)
        unfinished = it == self.sos_id
        for t in range(max_seq_len):
            logprobs, state = self._forward_step(it, state, input_features, x_mask)  # bs*vocab_size
            if sample_max:
                sample_logprobs, it = torch.max(logprobs.detach(), 1)
            else:
                it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
                sample_logprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
            it = it.view(-1).long()
            sample_logprobs = sample_logprobs.view(-1)

            seq_masks[:, t] = unfinished
            it = it * unfinished.type_as(it)  # bs
            seq[:, t] = it
            seq_logprobs[:, t] = sample_logprobs

            unfinished = unfinished * (it != self.eos_id)  # update if finished according to EOS
            if unfinished.sum() == 0:
                break

        return seq, seq_logprobs, seq_masks
    def sample_max_batch(self, input_features, x_mask, decoding_constraint=1):
        self.eval()

        if x_mask:
            max_seq_len = input_features.shape[-1] // self.channel_dim
            input_features = self.from_channel_emb(input_features.view(input_features.shape[0], max_seq_len, -1))
        else:
            max_seq_len = 21

        batch_size = input_features.size(0)
        seq = input_features.new_zeros((batch_size, max_seq_len), dtype=torch.long)
        state = self.init_hidden(input_features)
        last_word_id = torch.zeros(batch_size, dtype=torch.long)
        it = input_features.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id)
        unfinished = it == self.sos_id
        for t in range(max_seq_len):

            logprobs, state = self._forward_step(it, state, input_features, x_mask)

            logprobs[:,self.pad_id] += float('-inf')  # do not generate <PAD>, <SOS> and <UNK>
            logprobs[:,self.sos_id] += float('-inf')
            logprobs[:,self.unk_id] += float('-inf')
            if decoding_constraint:  # do not generate last step word
                for idxx, xxx in enumerate(last_word_id):
                    logprobs[idxx, xxx] += float('-inf')
            it = torch.max(logprobs,1)[1]
            it = it * unfinished.type_as(it)  # once eos, output zero.
            seq[:,t] = it
            last_word_id = it.clone()
            unfinished = unfinished * (it != self.eos_id)

        return seq

class Embeds(nn.Module):
    def __init__(self, vocab_size, num_hidden):
        super(Embeds, self).__init__()
        self.emb = nn.Embedding(vocab_size, num_hidden, padding_idx=0)  # learnable params, nn.Embedding是用來將一個數字變成一個指定維度的向量的
        #vocab.GloVe(name='6B', dim=50, cache='../Glove') This is optional.

    def __call__(self, inputs):
        return self.emb(inputs)
class LSTMEncoder(nn.Module):
    def __init__(self, channel_dim, embedds):
        super(LSTMEncoder, self).__init__()

        self.num_hidden = 512
        self.pad_id = 0
        self.word_embed_encoder = embedds
        self.lstm_encoder = nn.LSTM(input_size=self.num_hidden, hidden_size=self.num_hidden,
                                    num_layers=2, bidirectional=True, batch_first=True)
        self.to_chanenl_embedding = nn.Sequential(nn.Linear(2*self.num_hidden, 2*self.num_hidden), nn.ReLU(),
                                                    nn.Linear(2*self.num_hidden, channel_dim))
        
    def forward(self, x, len_batch):
        # x: [batch, T]
        self.lstm_encoder.flatten_parameters()
        word_embs = self.word_embed_encoder(x)
       
        word_embs_packed = pack_padded_sequence(word_embs, len_batch, enforce_sorted=True, batch_first=True)
        
        output, state = self.lstm_encoder(word_embs_packed)  # output is a packed seq
        (_data ,_len) = pad_packed_sequence(output, batch_first=True)
        forward_ = word_embs.new_zeros(x.size(0), self.num_hidden, dtype=torch.float)
        backward_ = word_embs.new_zeros(x.size(0), self.num_hidden, dtype=torch.float)
       
        for i in range(x.size(0)):
            forward_[i,:] = _data[i, _len[i]-1, :self.num_hidden]  # we take the last forward step
            backward_[i,:] = _data[i, 0, self.num_hidden:]  # and the first backward step
        embedding_channel = self.to_chanenl_embedding(torch.cat([forward_, backward_], dim=1))
        return embedding_channel, None  # src_mask is None

class LSTMDecoder(nn.Module, Decoder_Meta):
    def __init__(self, channel_dim, embedds, vocab_size):
        super(LSTMDecoder, self).__init__()
        self.num_hidden = 512
        self.vocab_size = vocab_size
        self.channel_dim = channel_dim
        self.pad_id, self.sos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
        self.word_embed_decoder = embedds
        self.from_channel_emb = nn.Linear(channel_dim, 2*self.num_hidden)
        self.lstmcell_decoder = nn.LSTMCell(input_size=self.num_hidden, hidden_size=self.num_hidden)
        self.linear_and_dropout_classifier_decoder = nn.Sequential(
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.Dropout(0.5), nn.Linear(self.num_hidden, self.vocab_size))
        
    def _forward_step(self, it, state, placeholder1, placeholder2):  # compatibility with Transformer backbone
        word_embs = self.word_embed_decoder(it)  # bs*word_emb
        _h, _c = self.lstmcell_decoder(word_embs, state)
        output = self.linear_and_dropout_classifier_decoder(_h)  # [bs, vocab_size]
        logprobs = F.log_softmax(output, dim=-1)  # [bs*vocab_size] In LSTM cell, we always run one step, for T times.
        return logprobs, (_h, _c)

    def forward_ce(self, input_features, gt_captions, src_mask=None, ss_prob=None):
        assert ss_prob is not None, 'must provide ss_prob'
        batch_size = gt_captions.size(0)
        state = self.init_hidden(input_features)
        outputs = []
        for i in range(gt_captions.size(1)):  # length of captions.
            if self.training and i >= 1 and ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = input_features.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < ss_prob
                it = gt_captions[:, i - 1].clone()  # teacher forcing
                if sample_mask.sum() != 0:
                    sample_ind = sample_mask.nonzero().view(-1)
                    prob_prev = outputs[i - 1].detach().exp()  # bs*vocab_size, fetch prev distribution
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                    it = input_features.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id) if i==0 else \
                        gt_captions[:, i-1].clone()  # it is the input of decoder.

            logprobs, state = self._forward_step(it, state, None, None)  # placeholders, for compatibility
            outputs.append(logprobs)

        outputs = torch.stack(outputs, dim=1)  # [bs, max_len, vocab_size]
        return outputs
    
    def init_hidden(self, input):
        x = self.from_channel_emb(input)
        return (x[:,:self.num_hidden],  x[:,self.num_hidden:]) # split into half


    def forward(self, x, placeholder):
        # x: [bs, T]
        src_mask = (x != self.pad_id).unsqueeze(1)
        x = self.word_embed_encoder(x)
        x = self.PE(x)  # positional embedding
        x = self.encoder(x, src_mask)  # [bs, T, d_model]
        embedding_channel = self.to_chanenl_embedding(x)
        embedding_channel = embedding_channel.view(embedding_channel.shape[0],-1)
        return embedding_channel, src_mask
class Normlize_tx:
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex
    def apply(self, _input):
        _dim = _input.shape[1]//2 if self._iscomplex else _input.shape[1]
        _norm = _dim**0.1 / torch.sqrt(torch.sum(_input**2, dim=1))
        return _input*_norm.view(-1,1)

class Channel:
    # returns the message when passed through a channel.
    # AGWN, Fading
    # Note that we need to make sure that the colle map will not change in this
    # step, thus we should not use *= and +=.
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex

    def ideal_channel(self, _input):
        return _input

    def awgn(self, _input, _snr):
        _std = (10**(-_snr/10.)/2)**0.5 if self._iscomplex else (10**(-_snr/10.))**0.5  # for complex signals.
        _input = _input + torch.randn_like(_input) * _std

        return _input
    
    def smaple_n_times(n, x):
        if n>1:
            x = x.unsqueeze(1) # Bx1x...
            x = x.expand(-1, n, *([-1]*len(x.shape[2:])))
            x = x.reshape(x.shape[0]*n, *x.shape[2:])
        return x

def quantize_and_convert_to_binary_with_integer_part(input_tensor, error_boundary):
    # Define quantization ranges and values
    quantization_ranges = [
                           (-2, -1.9678), (-1.9677, -1.9366), (-1.9365, -1.9053),(-1.9052, -1.8741), (-1.874, -1.8429), (-1.8428, -1.8117), (-1.8116, -1.7805), (-1.7804, -1.7492),(-1.7491, -1.718), (-1.7179, -1.6868), (-1.6867, -1.6556),(-1.6555, -1.6244), (-1.6243, -1.5932), (-1.5931, -1.5619), (-1.5618, -1.5307), (-1.5306, -1.4995), (-1.4994, -1.4683), (-1.4682, -1.4371), (-1.437, -1.4058),(-1.4057, -1.3746), (-1.3745, -1.3434), (-1.3433, -1.3122), (-1.3121, -1.281), (-1.2809, -1.2497), (-1.2496, -1.2185), (-1.2184, -1.1873), (-1.1872, -1.1561),(-1.156, -1.1249), (-1.1248, -1.0937), (-1.0936, -1.0624), (-1.0623, -1.0312), (-1.0311, -1.0), 
                            (-0.999, -0.9678), (-0.9677, -0.9366), (-0.9365, -0.9053),(-0.9052, -0.8741), (-0.874, -0.8429), (-0.8428, -0.8117), (-0.8116, -0.7805), (-0.7804, -0.7492),(-0.7491, -0.718), (-0.7179, -0.6868), (-0.6867, -0.6556),(-0.6555, -0.6244), (-0.6243, -0.5932), (-0.5931, -0.5619), (-0.5618, -0.5307), (-0.5306, -0.4995), (-0.4994, -0.4683), (-0.4682, -0.4371), (-0.437, -0.4058),(-0.4057, -0.3746), (-0.3745, -0.3434), (-0.3433, -0.3122), (-0.3121, -0.281), (-0.2809, -0.2497), (-0.2496, -0.2185), (-0.2184, -0.1873), (-0.1872, -0.1561),(-0.156, -0.1249), (-0.1248, -0.0937), (-0.0936, -0.0624), (-0.0623, -0.0312), (-0.0311, 0.0),    
                            (0.001, 0.0311), (0.0312, 0.0623), (0.0624, 0.0936),(0.0937, 0.1248), (0.1249, 0.156), (0.1561, 0.1872), (0.1873, 0.2184), (0.2185, 0.2497),(0.2498, 0.2809), (0.281, 0.3121), (0.3122, 0.3433),(0.3434, 0.3745), (0.3746, 0.4057), (0.4058, 0.437), (0.4371, 0.4682), (0.4683, 0.4994), (0.4995, 0.5306), (0.5307, 0.5618), (0.5619, 0.5931),(0.5932, 0.6243), (0.6244, 0.6555), (0.6556, 0.6867), (0.6868, 0.7179), (0.718, 0.7491), (0.7492, 0.7804), (0.7805, 0.8116), (0.8117, 0.8428),(0.8429, 0.874), (0.8741, 0.9052), (0.9053, 0.9365), (0.9366, 0.9677), (0.9678, 0.9999),    
                            (1.001, 1.0311), (1.0312, 1.0623), (1.0624, 1.0936),(1.0937, 1.1248), (1.1249, 1.156), (1.1561, 1.1872), (1.1873, 1.2184), (1.2185, 1.2497),(1.2498, 1.2809), (1.281, 1.3121), (1.3122, 1.3433),(1.3434, 1.3745), (1.3746, 1.4057), (1.4058, 1.437), (1.4371, 1.4682), (1.4683, 1.4994), (1.4995, 1.5306), (1.5307, 1.5618), (1.5619, 1.5931),(1.5932, 1.6243), (1.6244, 1.6555), (1.6556, 1.6867), (1.6868, 1.7179), (1.718, 1.7491), (1.7492, 1.7804), (1.7805, 1.8116), (1.8117, 1.8428),(1.8429, 1.874), (1.8741, 1.9052), (1.9053, 1.9365), (1.9366, 1.9677), (1.9678, 1.9999),  
       ]
    quantization_values = [-1.9833, -1.9522, -1.9209, -1.8897, -1.8584, -1.8273, -1.796, -1.7648, -1.7335, -1.7024, -1.6711, -1.64, -1.6087, -1.5775, -1.5462, -1.5151, -1.4839, -1.4526, -1.4214, -1.3901, -1.3589, -1.3277, -1.2965, -1.2653, -1.2341, -1.2029, -1.1716, -1.1404, -1.1092, -1.078, -1.0467, -1.0155,
                          -0.9833, -0.9522, -0.9209, -0.8897, -0.8584, -0.8273, -0.796, -0.7648, -0.7335, -0.7024, -0.6711, -0.64, -0.6087, -0.5775, -0.5462, -0.5151, -0.4839, -0.4526, -0.4214, -0.3901, -0.3589, -0.3277, -0.2965, -0.2653, -0.2341, -0.2029, -0.1716, -0.1404, -0.1092, -0.078, -0.0467, -0.0155,
                           0.0155, 0.0467, 0.078, 0.1092, 0.1404, 0.1716, 0.2029, 0.2341, 0.2653, 0.2965, 0.3277, 0.3589, 0.3901, 0.4214, 0.4526, 0.4839, 0.5151, 0.5462, 0.5775, 0.6087, 0.64, 0.6711, 0.7024, 0.7335, 0.7648, 0.796, 0.8273, 0.8584, 0.8897, 0.9209, 0.9522, 0.9833,
                            1.0155, 1.0467, 1.078, 1.1092, 1.1404, 1.1716, 1.2029, 1.2341, 1.2653, 1.2965, 1.3277, 1.3589, 1.3901, 1.4214, 1.4526, 1.4839, 1.5151, 1.5462, 1.5775, 1.6087, 1.64, 1.6711, 1.7024, 1.7335, 1.7648, 1.796, 1.8273, 1.8584, 1.8897, 1.9209, 1.9522, 1.9833,
                            ]
    # Quantize the tensor
    quantized_tensor = input_tensor.clone()
    for i in range(input_tensor.shape[0]):
        for j in range(input_tensor.shape[1]):
            value = input_tensor[i][j] + 0.9999  # Normalize the input to the range 0 to 1.9998
        
            for q_range, q_value in zip(quantization_ranges, quantization_values):
                lower_bound, upper_bound = q_range
                if lower_bound <= value < upper_bound:
                    quantized_tensor[i][j] = q_value
                    break
    print(quantized_tensor)
    # Convert to binary representation
    binary_tensor = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], 6, dtype=torch.int32)
    for i in range(quantized_tensor.shape[0]):
        for j in range(quantized_tensor.shape[1]):
            value = quantized_tensor[i][j]

            # Set the integer part bits (first bit)
            integer_part = int(value)
            binary_tensor[i, j, :0] = 1  

            # Set the integer part bits (6th to 10th bits)
            # For -0.9252, set to 0; for others set to 1
            if (value == 0.0155) or(value == 0.0467) or (value == 0.078) or (value == 0.1092) or (value == 0.1404) or(value == 0.1716) or (value == 0.2029) or (value == 0.2341) or(value == 0.2653) or (value == 0.2965) or (value == 0.3277) or (value == 0.3589) or(value == 0.3901) or (value == 0.4214) or (value == 0.4526)or(value == 0.4839) or (value == 0.5151) or (value == 0.5462) or (value == 0.5775) or(value == 0.6087) or (value == 0.64) or (value == 0.6711)or(value == 0.7024) or (value == 0.7335) or (value == 0.7648) or (value == 0.796) or(value == 0.8273) or (value == 0.8584) or (value == 0.8897)or(value == 0.9209) or (value == 0.9522) or (value == 0.9833): 
                binary_tensor[i, j, 0:1] = 0
            else:
                binary_tensor[i, j, 0:1] = 1

          

         

            # Set the fractional part bits (second and third bits)
            fractional_part = value - integer_part
            if 0 < fractional_part < 0.0311:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 0
            elif 0.0312 < fractional_part < 0.0623:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 1
            elif 0.0624 < fractional_part < 0.0936:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 0
            elif 0.0937 < fractional_part < 0.1248:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 1
            elif 0.1249 < fractional_part < 0.156:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 0
            elif 0.1561 < fractional_part < 0.1872:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 1
            elif 0.1873 < fractional_part < 0.2184:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 0
            elif 0.2185 < fractional_part < 0.2497:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 1
            elif 0.2498 < fractional_part < 0.2809:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 0
            elif 0.281 < fractional_part < 0.3121:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 1
            elif 0.3122 < fractional_part < 0.3433:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 0
            elif 0.3434 < fractional_part < 0.3745:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 1
            elif 0.3746 < fractional_part < 0.4057:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 0
            elif 0.4058 < fractional_part < 0.437:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 1
            elif 0.4371 < fractional_part < 0.4682:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 0
            elif 0.4683 < fractional_part < 0.4994:
                binary_tensor[i, j, 1] = 0
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 1
            elif 0.4995 < fractional_part < 0.5306:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 0
            elif 0.5307 < fractional_part < 0.5618:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 1
            elif 0.5619 < fractional_part < 0.5931:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 0
            elif 0.5932 < fractional_part < 0.6243:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 1
            elif 0.6244 < fractional_part < 0.6555:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 0
            elif 0.6556 < fractional_part < 0.6867:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 1
            elif 0.6868 < fractional_part < 0.7179:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 0
            elif 0.718 < fractional_part < 0.7491:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 0
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 1
            elif 0.7492 < fractional_part < 0.7804:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 0
            elif 0.7805 < fractional_part < 0.8116:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 1
            elif 0.8117 < fractional_part < 0.8428:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 0
            elif 0.8429< fractional_part < 0.874:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 1
            elif 0.8741 < fractional_part < 0.9052:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 0
            elif 0.9053 < fractional_part < 0.9365:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 0
                binary_tensor[i, j, 5] = 1
            elif 0.9366 < fractional_part < 0.9677:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 0

            else:
                binary_tensor[i, j, 1] = 1
                binary_tensor[i, j, 2] = 1
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 1
                binary_tensor[i, j, 5] = 1
    return binary_tensor

def convert_binary_to_float(binary_tensor, quantization_values):
    # Tensor to hold the floating point values
    restored_tensor = torch.zeros(binary_tensor.shape[0], binary_tensor.shape[1], dtype=torch.float32)

    # Dictionary to map the binary representation of the fractional part to its float value
    fractional_mapping = {
        (0, 0, 0, 0, 0): 0.0155,
        (0, 0, 0, 0, 1): 0.0467,
        (0, 0, 0, 1, 0): 0.078,
        (0, 0, 0, 1, 1): 0.1092,
        (0, 0, 1, 0, 0): 0.1404,
        (0, 0, 1, 0, 1): 0.1716,
        (0, 0, 1, 1, 0): 0.2029,
        (0, 0, 1, 1, 1): 0.2341,
        (0, 1, 0, 0, 0): 0.2653,
        (0, 1, 0, 0, 1): 0.2965,
        (0, 1, 0, 1, 0): 0.3277,
        (0, 1, 0, 1, 1): 0.3589,
        (0, 1, 1, 0, 0): 0.3901,
        (0, 1, 1, 0, 1): 0.4214,
        (0, 1, 1, 1, 0): 0.4526,
        (0, 1, 1, 1, 1): 0.4839,
        (1, 0, 0, 0, 0): 0.5151,
        (1, 0, 0, 0, 1): 0.5462,
        (1, 0, 0, 1, 0): 0.5775,
        (1, 0, 0, 1, 1): 0.6087,
        (1, 0, 1, 0, 0): 0.64,
        (1, 0, 1, 0, 1): 0.6711,
        (1, 0, 1, 1, 0): 0.7024,
        (1, 0, 1, 1, 1): 0.7335,
        (1, 1, 0, 0, 0): 0.7648,
        (1, 1, 0, 0, 1): 0.796,
        (1, 1, 0, 1, 0): 0.8273,
        (1, 1, 0, 1, 1): 0.8584,
        (1, 1, 1, 0, 0): 0.8897,
        (1, 1, 1, 0, 1): 0.9209,
        (1, 1, 1, 1, 0): 0.9522,
        (1, 1, 1, 1, 1): 0.9833,
 


    }

    for i in range(binary_tensor.shape[0]):
        for j in range(binary_tensor.shape[1]):
            # Extracting individual bits from the binary representation
            integer_bits = binary_tensor[i, j, :1]
            fractional_bits = binary_tensor[i, j, 1:]

            # Determine the integer part: 1 if the majority of the next 5 bits are 1
            integer_part = 1 if integer_bits.sum() > 0 else 0

            # Convert the tuple of fractional bits to a float using the mapping
            fractional_value = fractional_mapping[tuple(fractional_bits.tolist())]

            # Combine to form the original value
            restored_value = integer_part + fractional_value

            restored_tensor[i][j] = restored_value

    return restored_tensor

# Modulation

# Function to convert a bit sequence to a decimal number
def bits_to_decimal(bits):
    # Using PyTorch operations to convert binary to decimal
    powers = torch.tensor([2**i for i in range(bits.size(-1)-1, -1, -1)])
    return torch.sum(bits * powers, dim=-1)

# Function to map decimal numbers to 16QAM constellation points
def decimal_to_64qam(decimal):
    # Create a simple mapping from decimal numbers to constellation points
    # This is not a standard mapping, just for demonstration
    mapping = torch.tensor([
        (-7, -7), (-7, -5), (-7, -3), (-7, -1), (-7, 1), (-7, 3), (-7, 5), (-7, 7),
     (-5, -7), (-5, -5), (-5, -3), (-5, -1),  (-5, 1), (-5, 3), (-5, 5), (-5, 7),
      (-3, -7), (-3, -5), (-3, -3), (-3, -1), (-3, 1), (-3, 3), (-3, 5), (-3, 7),
     (-1, -7), (-1, -5), (-1, -3), (-1, -1),  (-1, 1), (-1, 3), (-1, 5), (-1, 7),
     (1, -7), (1, -5), (1, -3), (1, -1), (1, 1), (1, 3), (1, 5), (1, 7),
     (3, -7), (3, -5), (3, -3), (3, -1),  (3, 1), (3, 3), (3, 5), (3, 7),
     (5, -7), (5, -5), (5, -3), (5, -1), (5, 1), (5, 3), (5, 5), (5, 7),
     (7, -7), (7, -5), (7, -3), (7, -1),  (7, 1), (7, 3), (7, 5), (7, 7),
    ])
    return mapping[decimal]

def awgn(_input, _snr, _iscomplex=False):
    _std = (10**(-_snr/10.)/2)**0.5 if _iscomplex else (10**(-_snr/10.))**0.5  # for complex signals.
    noise = torch.randn(_input.size()) * _std
    return _input + noise


# Function to find the nearest constellation point
def find_nearest_constellation_point(point):
    min_distance = float('inf')
    nearest_point = None

    inverse_mapping = {  
             (-7, -7):0, (-7, -5):1, (-7, -3):2, (-7, -1):3, (-7, 1):4, (-7, 3):5, (-7, 5):6, (-7, 7):7,
     (-5, -7):8, (-5, -5):9, (-5, -3):10, (-5, -1):11,  (-5, 1):12, (-5, 3):13, (-5, 5):14, (-5, 7):15,
      (-3, -7):16, (-3, -5):17, (-3, -3):18, (-3, -1):19, (-3, 1):20, (-3, 3):21, (-3, 5):22, (-3, 7):23,
     (-1, -7):24, (-1, -5):25, (-1, -3):26, (-1, -1):27,  (-1, 1):28, (-1, 3):29, (-1, 5):30, (-1, 7):31,
     (1, -7):32, (1, -5):33, (1, -3):34, (1, -1):35, (1, 1):36, (1, 3):37, (1, 5):38, (1, 7):39,
     (3, -7):40, (3, -5):41, (3, -3):42, (3, -1):43,  (3, 1):44, (3, 3):45, (3, 5):46, (3, 7):47,
     (5, -7):48, (5, -5):49, (5, -3):50, (5, -1):51, (5, 1):52, (5, 3):53, (5, 5):54, (5, 7):55,
     (7, -7):56, (7, -5):57, (7, -3):58, (7, -1):59,  (7, 1):60, (7, 3):61, (7, 5):62, (7, 7):63,
}

    for constellation_point, decimal in inverse_mapping.items():
        distance = (constellation_point[0] - point[0]) ** 2 + (constellation_point[1] - point[1]) ** 2
        if distance < min_distance:
            min_distance = distance
            nearest_point = decimal
    return nearest_point

# Function to convert decimal numbers back to bit sequences
def decimal_to_bits(decimal):
    # Convert decimal number back to a 4-bit binary representation
    return torch.tensor([int(bit) for bit in format(decimal, '06b')])


device = torch.device("cpu:0")
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
embeds_shared = Embeds(vocab_size=24, num_hidden=512).to(device)
encoder = LSTMEncoder(channel_dim=channel_dim, embedds=embeds_shared).to(device)
decoder = LSTMDecoder(channel_dim=channel_dim, embedds=embeds_shared, vocab_size = 24).to(device)

encoder = encoder.eval()
decoder = decoder.eval()
embeds_shared = embeds_shared.eval()


normlize_layer = Normlize_tx(_iscomplex=_iscomplex)
channel = Channel(_iscomplex=_iscomplex)

def do_test(input_data, encoder, decoder, normlize_layer, channel, len_batch):
    total_errors_accumulated = 0
    total_bits_accumulated = 0
    with torch.no_grad():
        print("input_data :")
        print(input_data)
        output, _ = encoder(input_data, len_batch)
        print("encoder :")
        print(output)
        output = normlize_layer.apply(output)
        print("normlize_layer :")
        print(output)
        
        print("Quantized :")
        error_boundary = 0.0624

        pred1 = quantize_and_convert_to_binary_with_integer_part(output, error_boundary)
        
        print(f"Binarizer :\n{pred1}")
        bit_sequences = pred1.view(-1, 6)
        
        # Convert bit sequences to decimal
        decimal_numbers = bits_to_decimal(bit_sequences)

        # Map decimal numbers to 16QAM constellation points
        modulated_symbols = decimal_to_64qam(decimal_numbers)
        print("Modulation :")
        print(modulated_symbols)

        # Adding AWGN to the modulated symbols
        noisy_symbols = awgn(modulated_symbols.float(), _snr, is_complex)
     
        print("Channel noise :")
        print(noisy_symbols)

        # Demodulate the noisy symbols
        demodulated_decimals = torch.tensor([find_nearest_constellation_point(symbol) for symbol in noisy_symbols])
        print("Demodulation :")
        print(demodulated_decimals)

        # Convert demodulated decimals back to bit sequences
        demodulated_bit_sequences = torch.stack([decimal_to_bits(decimal) for decimal in demodulated_decimals])
        demodulated_bit_sequences = demodulated_bit_sequences.unsqueeze(0)
        print("Debinarizer :")
        print(demodulated_bit_sequences)
       


        quantization_values = [0.0155, 0.0467, 0.078, 0.1092, 0.1404, 0.1716, 0.2029, 0.2341, 0.2653, 0.2965, 0.3277, 0.3589, 0.3901, 0.4214, 0.4526, 0.4839, 0.5151, 0.5462, 0.5775, 0.6087, 0.64, 0.6711, 0.7024, 0.7335, 0.7648, 0.796, 0.8273, 0.8584, 0.8897, 0.9209, 0.9522, 0.9833,
                            1.0155, 1.0467, 1.078, 1.1092, 1.1404, 1.1716, 1.2029, 1.2341, 1.2653, 1.2965, 1.3277, 1.3589, 1.3901, 1.4214, 1.4526, 1.4839, 1.5151, 1.5462, 1.5775, 1.6087, 1.64, 1.6711, 1.7024, 1.7335, 1.7648, 1.796, 1.8273, 1.8584, 1.8897, 1.9209, 1.9522, 1.9833,
                         ]
        pred8 = convert_binary_to_float(demodulated_bit_sequences, quantization_values)
        print("Dequantized :")
        # print(pred8)
        
        pred8 = pred8 - 0.9999
        
        print(pred8)
     
        output = decoder.sample_max_batch(pred8, None)
        print("Decoder :")
        print(output)
    
    return output
SemanticRL_example = [          
                     
                 #       'downlink information and vrb will map to prb and retransmission and a serving cell received',
                         'planets engage in an eternal graceful dance around sun rise'
                        # 'moon casts gentle soft glow across dark darkness took blue',
                        #  'from ue to gnb'
                      ]
#input_str = "['this message will send downlink information'\n'vrb to prb is an interleaved mapping way' ]"
input_str = "['planets engage in an eternal graceful dance around sun rise']"
# input_str = "['moon casts gentle soft glow across dark darkness took blue']"


processed_str = input_str.strip("[]").replace("'", "")
print('--------------Reference sentence---------------')
print(processed_str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_pathCE", type=str, default='./ckpt_AWGN_CE')
    parser.add_argument("--ckpt_pathRL", type=str, default='./ckpt_AWGN_RL')  # or './ckpt_AWGN_RL'
    args = parser.parse_args()

    # dict_train = pickle.load(open('./1.train_dict.pkl', 'rb'))
    dict_train = pickle.load(open('/home/eric/mwnl/train_dict.pkl', 'rb'))
    rev_dict = {vv: kk for kk, vv in dict_train.items()}

    success_count = 0
    failure_count = 0

    # total_errors_accumulated = 0
    # total_bits_accumulated = 0
    with open('log.txt', 'w') as log_file:
        for input_str in SemanticRL_example:

            input_vector = [dict_train[x] for x in input_str.split(' ')] + [2]
            input_len = len(input_vector)
            input_vector = torch.tensor(input_vector)

      
    
            for ckpt_dir in [args.ckpt_pathCE]:#, args.ckpt_pathRL
                model_name = os.path.basename(ckpt_dir)
    
                encoder.load_state_dict(torch.load('/home/eric/mwnl/ckpt_AWGN_CE/encoder_epoch99.pth', map_location='cpu'))
                decoder.load_state_dict(torch.load('/home/eric/mwnl/ckpt_AWGN_CE/decoder_epoch99.pth', map_location='cpu'))
                embeds_shared.load_state_dict(torch.load('/home/eric/mwnl/ckpt_AWGN_CE/embeds_shared_epoch99.pth',  map_location='cpu'))

                for _ in range(20000):
                    output = do_test(input_vector.unsqueeze(0), encoder, decoder, normlize_layer, channel,
                        len_batch=torch.tensor(input_len).view(-1, ))
                    output = output.cpu().numpy()[0]
                    res = ' '.join(rev_dict[x] for x in output if x!=0 and x!=2)  # remove 'PAD' and 'EOS'

                    print('--------------Candidate sentence---------------')
                    print('{}'.format(res))
                  
                    print('-----------------------------------------------')
                    print('------------------Comparison-------------------')
                    print('-----------------------------------------------')

                    sent_a_reference = 'planets engage in an eternal graceful dance around sun rise'.split()
                    print('sent_a_reference_sentence = {} '.format(sent_a_reference))
                    
                    sent_b_reference = 'moon casts gentle soft glow across dark darkness took blue'.split()
                    print('sent_b_reference_sentence = {} '.format(sent_b_reference))
                    

                    sent_a_candidate = ' {} '.format(res).split()
                    print('sent_a_candidate_sentence = {} '.format(sent_a_candidate))

                    print('-----------------------------------------------')
                    print('-----------------BLEU-4 score------------------')
                    print('-----------------------------------------------')

                    smoothie = SmoothingFunction().method2
                    bleu1= bleu([sent_a_reference], sent_a_candidate, smoothing_function=smoothie)
                    smoothie = SmoothingFunction().method2
                    bleu2= bleu([sent_b_reference], sent_a_candidate, smoothing_function=smoothie)

                    print('bleu score 1 (sent_a_reference_sentence, sent_a_candidate_sentence)= {} '.format(bleu1))

                    if ( bleu1 > bleu2):
                        print('bleu score 1 > bleu score 2 = {} '.format(bleu1))
                        print('sent_a_reference_sentence = {} '.format(sent_a_reference))
                        print('sent_a_candidate_sentence = {} '.format(sent_a_candidate))
                        print('Confirmation sent_a_reference_sentence sent message \n')
                    
        
                        success_count += 1
           
                    
                    else :
                        print('bleu score 2 > bleu score 1 = {} '.format(bleu2))
                        failure_count += 1
         

               
    # Print the success and failure counts
    print('Success count:', success_count)
    print('Failure count:', failure_count)
