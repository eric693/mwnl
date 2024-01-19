import os, platform, json, time, pickle, sys, argparse, random, datetime
import torch
from math import log
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
import numpy as np

_snr = 10
_iscomplex = False
channel_dim = 2

class Decoder_Meta():
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
        self.emb = nn.Embedding(vocab_size, num_hidden, padding_idx=0) 
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

        outputs = torch.stack(outputs, dim=1) 
        return outputs
    
    def init_hidden(self, input):
        x = self.from_channel_emb(input)
        return (x[:,:self.num_hidden],  x[:,self.num_hidden:]) 

    def forward(self, x, placeholder):
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
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex

    def awgn(self, _input, _snr):
        _std = (10**(-_snr/10.)/2)**0.5 if self._iscomplex else (10**(-_snr/10.))**0.5  # for complex signals.
        _input = _input + torch.randn_like(_input) * _std
        return _input

def quantize_and_convert_to_binary_with_integer_part(input_tensor, error_boundary):
    quantization_ranges = [
                           (-2.0, -1.751), (-1.75, -1.51), (-1.5, -1.251), (-1.25, -1.001),
                           (-1.0, -0.751), (-0.75, -0.51), (-0.5, -0.251), (-0.25, -0.001),
                           (0.0, 0.249), (0.25, 0.499), (0.5, 0.749), (0.75, 0.99), 
                           (1.0, 1.249), (1.25, 1.499), (1.5, 1.749), (1.75, 1.99)]
    quantization_values = [-1.875, -1.625, -1.375, -1.125, -0.875, -0.625, -0.375, -0.125,
                           0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875]

    quantized_tensor = input_tensor.clone()
    for i in range(input_tensor.shape[0]):
        for j in range(input_tensor.shape[1]):
            value = input_tensor[i][j] + 0.9999  
        
            for q_range, q_value in zip(quantization_ranges, quantization_values):
                lower_bound, upper_bound = q_range
                if lower_bound <= value < upper_bound:
                    quantized_tensor[i][j] = q_value
                    break
    print(quantized_tensor)

    binary_tensor = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], 5, dtype=torch.int32)
    for i in range(quantized_tensor.shape[0]):
        for j in range(quantized_tensor.shape[1]):
            value = quantized_tensor[i][j]

            integer_part = int(value)
            binary_tensor[i, j, :0] = 1  

            if (value == 0.125) or(value == 0.375) or (value == 0.625) or (value == 0.875):  # Quantized value for -0.9252
                binary_tensor[i, j, 0:3] = 0
            else:
                binary_tensor[i, j, 0:3] = 1
   
            if (value == 1.125) or (value == 1.375) or (value == 1.625) or (value == 1.875):
                binary_tensor[i, j, 0:3] = 1 
            else:
                binary_tensor[i, j, 0:3] = 0  

            fractional_part = value - integer_part
            if fractional_part < 0.25:
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 0
            elif fractional_part < 0.5:
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 1
            elif fractional_part < 0.75:
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 0
            else:
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 1

    return binary_tensor

def convert_binary_to_float(binary_tensor, quantization_values):
    restored_tensor = torch.zeros(binary_tensor.shape[0], binary_tensor.shape[1], dtype=torch.float32)
    fractional_mapping = {
        (0, 0): 0.125,
        (0, 1): 0.375,
        (1, 0): 0.625,
        (1, 1): 0.875
    }
    return restored_tensor
def convert_binary_to_float_v1(binary_tensor):

    restored_values = torch.zeros(binary_tensor.shape[1], dtype=torch.float32)
    fractional_mapping_v1 = {

(1, 1): 1.625,
(0, 1): 1.625,
(1, 0): 1.625,
(0, 0): 0.375,
    }
    fractional_mapping_v2= {

(1, 1): 0.125,
(1, 0): 0.125,
(0, 1): 0.125,
(0, 0): 1.875,

    }

    for i, mapping in enumerate([fractional_mapping_v1, fractional_mapping_v2]):
        binary_bits = tuple(binary_tensor[0][i])  # Accessing the row for 4-bit binary
        binary_bits_int = tuple(bit.item() for bit in binary_bits)
        restored_values[i] = mapping[binary_bits_int]
    return restored_values

device = torch.device("cpu:0")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
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
        error_boundary = 0.125 
        pred1 = quantize_and_convert_to_binary_with_integer_part(output, error_boundary)
        replaced_tensor = torch.tensor([[[0, 0], [0, 0]]])

        print(replaced_tensor)
        print("Channel noise :")
        BER_target = 0

        error_mask = torch.rand(replaced_tensor.shape) < BER_target
        received_pred1 = replaced_tensor ^ error_mask.long()
        num_errors = torch.sum(replaced_tensor != received_pred1)
        total_bits_accumulated += replaced_tensor.numel()
        total_errors_accumulated += num_errors.item()
        
        print(received_pred1)
        pred8 = convert_binary_to_float_v1(received_pred1)
        
        print("bit vector convert float_point :")

        pred8 = pred8.unsqueeze(0)
        print(pred8)
        pred8 = pred8 - 0.9999

        print("Dequantized :")
        print(pred8)
     
        output = decoder.sample_max_batch(pred8, None)
        print("decoder :")
        print(output)
        calculated_BER = total_errors_accumulated / total_bits_accumulated if total_bits_accumulated > 0 else 0
    return output
SemanticRL_example = [          
                         'planets engage in an eternal graceful dance around sun rise',
                      ]
input_str = "['planets engage in an eternal graceful dance around sun rise']"

processed_str = input_str.strip("[]").replace("'", "")
print('--------------Reference sentence---------------')
print(processed_str)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_pathCE", type=str, default='./ckpt_AWGN_CE')
    # parser.add_argument("--ckpt_pathRL", type=str, default='./ckpt_AWGN_RL')  # or './ckpt_AWGN_RL'
    args = parser.parse_args()

    dict_train = pickle.load(open('/home/eric/mwnl/train_dict.pkl', 'rb'))
    rev_dict = {vv: kk for kk, vv in dict_train.items()}

    for input_str in SemanticRL_example:

        input_vector = [dict_train[x] for x in input_str.split(' ')] + [2]
        input_len = len(input_vector)
        input_vector = torch.tensor(input_vector)

        for ckpt_dir in [args.ckpt_pathCE]:#, args.ckpt_pathRL
            model_name = os.path.basename(ckpt_dir)

            encoder.load_state_dict(torch.load('/home/eric/mwnl/ckpt_AWGN_CE/encoder_epoch99.pth', map_location='cpu'))
            decoder.load_state_dict(torch.load('/home/eric/mwnl/ckpt_AWGN_CE/decoder_epoch99.pth', map_location='cpu'))
            embeds_shared.load_state_dict(torch.load('/home/eric/mwnl/ckpt_AWGN_CE/embeds_shared_epoch99.pth',  map_location='cpu'))

            start_time = time.time()
            for _ in range(1):
                    
                output = do_test(input_vector.unsqueeze(0), encoder, decoder, normlize_layer, channel,
                    len_batch=torch.tensor(input_len).view(-1, ))
                
                output = output.cpu().numpy()[0]
                res = ' '.join(rev_dict[x] for x in output if x!=0 and x!=2)  # remove 'PAD' and 'EOS'
            end_time = time.time()
            duration_in_milliseconds = (end_time - start_time) 
            print("Python run time:")
            print(duration_in_milliseconds)
