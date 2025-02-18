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


_snr = 10
_iscomplex = True
channel_dim = 8

##############

def collate_func(in_data):
    batch_tensor, batch_len = list(zip(*(sorted(in_data, key=lambda s:-s[1]))))
    return torch.stack(batch_tensor, dim=0), batch_len

##################

BeamCandidate = namedtuple('BeamCandidate',
                           ['state', 'log_prob_sum', 'log_prob_seq', 'last_word_id', 'word_id_seq'])
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

    
##################
class Normlize_tx:
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex
    def apply(self, _input):
        _dim = _input.shape[1]//2 if self._iscomplex else _input.shape[1]
        _norm = _dim**0.5 / torch.sqrt(torch.sum(_input**2, dim=1))
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
        #print(_std)
        #print(_input)
        return _input
    
    def smaple_n_times(n, x):
        if n>1:
            x = x.unsqueeze(1) # Bx1x...
            x = x.expand(-1, n, *([-1]*len(x.shape[2:])))
            x = x.reshape(x.shape[0]*n, *x.shape[2:])
        return x



class TransformerEncoder(nn.Module):
    def __init__(self, channel_dim, embedds):
        super(TransformerEncoder, self).__init__()
        self.num_hidden = 512
        self.channel_dim = channel_dim
        self.pad_id = 0
        self.word_embed_encoder = embedds
        self.encoder = make_encoder(N=3, d_model=512, d_ff=256, h=4, dropout=0.1)
        self.PE = PositionalEncoding(d_model=512, dropout=0.1)
        self.to_chanenl_embedding = nn.Sequential(nn.Linear(self.num_hidden, self.num_hidden), nn.ReLU(),
                                                    nn.Linear(self.num_hidden, channel_dim))

    def forward(self, x, placeholder):
        # x: [bs, T]
        src_mask = (x != self.pad_id).unsqueeze(1)
        x = self.word_embed_encoder(x)
        x = self.PE(x)  # positional embedding
        x = self.encoder(x, src_mask)  # [bs, T, d_model]
        embedding_channel = self.to_chanenl_embedding(x)
        embedding_channel = embedding_channel.view(embedding_channel.shape[0],-1)
        return embedding_channel, src_mask


class TransformerDecoder(nn.Module, Decoder_Meta):
    def __init__(self, channel_dim, embedds, vocab_size):
        super(TransformerDecoder, self).__init__()
        self.num_hidden = 512
        self.channel_dim = channel_dim
        self.vocab_size = vocab_size
        self.pad_id, self.sos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
        self.word_embed_decoder = embedds
        self.from_channel_emb = nn.Linear(channel_dim, self.num_hidden)
        self.decoder = make_decoder(N=3, d_model=512, d_ff=256, h=4, dropout=0.1)
        self.PE = PositionalEncoding(d_model=512, dropout=0.1)
        self.linear_and_dropout_classifier_decoder = nn.Sequential(
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.Dropout(0.5), nn.Linear(self.num_hidden, self.vocab_size))

    @staticmethod
    def subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def forward_ce(self, x_memory, y, src_mask, ss_prob=None):
        y = y[:,:-1] # teacher forcing. We feed Transformer model with [<SOS> w_1 w_2 w_3 ...]
        x_memory = x_memory.view(x_memory.shape[0], -1, self.channel_dim)
        y_mask = (y != self.pad_id).unsqueeze(-2) # [bs, 1, T]
        y_mask = (y_mask & smaple_n_times(y_mask.shape[0], self.subsequent_mask(y.size(-1))).type_as(y_mask)) # [bs, T, T]
        x_memory = self.from_channel_emb(x_memory)
        y = self.word_embed_decoder(y)
        y = self.PE(y)
        y = self.decoder(y, x_memory, src_mask, y_mask)
        output = self.linear_and_dropout_classifier_decoder(y)  # [bs, T, vocab_size]
        logprobs = F.log_softmax(output, dim=-1)  # [bs, T, vocab_size]
        return logprobs

    def init_hidden(self, input):  # Transformer state.
        return []

    def _forward_step(self, it, state, x_memory, x_mask):
        if len(state) == 0: ys = it.unsqueeze(1)
        else: ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)  # update the recurrent state $s$
        ys_nn = self.word_embed_decoder(ys)
        ys_nn = self.PE(ys_nn)
        out = self.decoder(ys_nn, x_memory, x_mask, smaple_n_times(x_mask.shape[0], self.subsequent_mask(ys.size(1))).to(x_memory.device))
        output = self.linear_and_dropout_classifier_decoder(out[:,-1])  # we only care the output at present time step
        logprobs = F.log_softmax(output, dim=-1)  # [bs*M, vocab_size], where M is the number of parallel samples
        return logprobs, [ys.unsqueeze(0)]  # output, state

    # RL can also get integrated with teacher forcing. This works fine with Transformer backbone.
    def forward_rl_ssprob(self, input_features, gt_captions, sample_max=None, multiple_sample=5, x_mask=None, ss_prob=1):

        max_seq_len = input_features.shape[-1]//self.channel_dim

        if x_mask is not None: # Transformer
            input_features = smaple_n_times(multiple_sample, input_features.view(input_features.shape[0], max_seq_len, -1))
            input_features = self.from_channel_emb(input_features)
            x_mask = smaple_n_times(multiple_sample, x_mask)
        else: # LSTM
            input_features = smaple_n_times(multiple_sample, input_features)


        batch_size = input_features.size(0)
        state = self.init_hidden(input_features)

        seq = input_features.new_zeros((batch_size, max_seq_len), dtype=torch.long)
        seq_logprobs = input_features.new_zeros((batch_size, max_seq_len))
        seq_masks = input_features.new_zeros((batch_size, max_seq_len))
        it = input_features.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id)  # t=0
        unfinished = it == self.sos_id  # a flag indicates where the decoding process has finished
        for t in range(max_seq_len):
            sample_prob = input_features.new(batch_size).uniform_(0, 1)
            sample_mask = sample_prob < ss_prob
            it_gt = smaple_n_times(multiple_sample, gt_captions[:, t].clone().view(-1,1)).view(-1) if t>=1 else it

            sample_ind = sample_mask.nonzero().view(-1)
            it_mixed = it.index_copy(0, sample_ind, it_gt.index_select(0, sample_ind)) # get

            logprobs, state = self._forward_step(it_mixed, state, input_features, x_mask)
            if sample_max:
                sample_logprobs, it = torch.max(logprobs.detach(), 1)
            else:
                it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
                sample_logprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
            it = it.view(-1).long()
            sample_logprobs = sample_logprobs.view(-1)

            seq_masks[:, t] = unfinished
            it = it * unfinished.type_as(it)  # once faced with <EOS>, the rest part shou always be zero.
            seq[:, t] = it
            seq_logprobs[:, t] = sample_logprobs

            unfinished = unfinished * (it != self.eos_id)  # update if finished according to <EOS>
            if unfinished.sum() == 0:
                break

        return seq, seq_logprobs, seq_masks


if __name__ =='__main__':

    is_complex = False
    n = Normlize_tx(is_complex)
    x = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], [18., 2., 3., 4., 5., 6., 7., 8., 9., 10.]])
    y = n.apply(x)
    #print(y)
    #for i in range(x.shape[1]//2):
    #    print(y[:,i], y[:,5+i])

    c = Channel(is_complex)
    # x = torch.ones(2,4)
    z = c.awgn(y,10)
    #print(z)




def float_to_fixed_point_sign_5_5_6(x):

    sign_bits = 5
    int_bits = 5
    frac_bits = 6
    
  
    max_val = 2 ** (int_bits + frac_bits - 1) - 2 ** frac_bits
    min_val = -2 ** (int_bits + frac_bits - 1)


    x = torch.clamp(x, min_val, max_val)

    
    x_fixed = x * 2 ** frac_bits


    x_fixed = x_fixed.long()

  
    x_fixed = x_fixed.view(-1, 1)


    sign = torch.sign(x_fixed)


    int_part = torch.div(torch.abs(x_fixed), 2 ** frac_bits, rounding_mode='trunc')


    frac_part = torch.abs(x_fixed) % 2 ** frac_bits


    fixed_point = torch.cat([sign, int_part, frac_part], dim=1)


    fixed_point_bin = torch.zeros((fixed_point.shape[0], sign_bits + int_bits + frac_bits), dtype=torch.int)


    for i in range(sign_bits):
        fixed_point_bin[:, i] = (fixed_point[:, 0] < 0).long()


    if int_bits >= 5:
        fixed_point_bin[:, sign_bits:sign_bits+int_bits] = torch.abs(int_part)  # Repeat removed
    else:
        fixed_point_bin[:, sign_bits:sign_bits+int_bits] = torch.abs(int_part)


    for i in range(frac_bits):
        fixed_point_bin[:, sign_bits+int_bits+i] = torch.div(fixed_point[:, 2], 2 ** (frac_bits - 1 - i), rounding_mode='trunc') % 2

    return fixed_point_bin

def flip_bits(x, n):
    # Flatten the tensor into a 1D vector and randomly choose n indices
    flat_x = x.flatten()
    num_bits = len(flat_x)
    indices = random.sample(range(num_bits), n)
    
    # Flip the bits at the chosen indices
    flipped_bits = torch.zeros(num_bits, dtype=torch.bool)
    flipped_bits[indices] = True
    flipped_x = torch.where(flipped_bits, 1-flat_x, flat_x)
    
    # Reshape the flattened tensor back into its original shape
    return flipped_x.reshape(x.shape)

def fixed_point_to_float_sign_5_5_6(x_fixed, int_bits, frac_bits):
   
    x_fixed = x_fixed.view(-1, int_bits + frac_bits + 5)


    sign = torch.where(torch.sum(x_fixed[:, :5], dim=1) > 2, -1, 1)


    int_part = x_fixed[:, 5:5+int_bits]
    int_part = torch.where(torch.sum(int_part, dim=1) > 2, 1, 0)


    frac_part = x_fixed[:, 5+int_bits:5+int_bits+frac_bits]
    frac_part = torch.sum(frac_part * (1 / (2 ** torch.arange(1, frac_bits + 1))), dim=1)


    x = int_part + frac_part


    x = x * sign.float()

    return x.view(1, -1)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu:0")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
embeds_shared = Embeds(vocab_size=162, num_hidden=512).to(device)
encoder = LSTMEncoder(channel_dim=channel_dim, embedds=embeds_shared).to(device)
decoder = LSTMDecoder(channel_dim=channel_dim, embedds=embeds_shared, vocab_size = 162).to(device)

encoder = encoder.eval()
decoder = decoder.eval()
embeds_shared = embeds_shared.eval()


normlize_layer = Normlize_tx(_iscomplex=_iscomplex)
channel = Channel(_iscomplex=_iscomplex)

def do_test(input_data, encoder, decoder, normlize_layer, channel, len_batch):

    with torch.no_grad():
       
       
        print("Input_data :")
        print(input_data)
       
        output, _ = encoder(input_data, len_batch)
        print("Encoder :")
        print(output)
       
        output = normlize_layer.apply(output)
        print("Normlize_layer :")
        print(output)
       

        print("Float_point convert bit vector :")
        pred1 = float_to_fixed_point_sign_5_5_6(output)
        print(pred1) #fixed_point
       
    
        flipped = flip_bits(pred1, n=0)
        print(flipped)

        pred8 = fixed_point_to_float_sign_5_5_6(flipped, int_bits=5, frac_bits=6)
        print(pred8)
        output = decoder.sample_max_batch(pred8, None)#len_batch
        
        print("decoder :")
        print(output)

    return output
SemanticRL_example = [          
                 #       'downlink information and vrb will map to prb and retransmission and a serving cell received',
                         'planets planets engage engage in in an an eternal eternal'
                       #  'vrb directly mapped to prb',
                        #  'from ue to gnb'
                      ]
#input_str = "['this message will send downlink information'\n'vrb to prb is an interleaved mapping way' ]"
input_str = "['Planets Planets engage engage in in an an eternal eternal']"
#input_str = "['downlink information and vrb will map to prb and retransmission and a serving cell received']"


processed_str = input_str.strip("[]").replace("'", "")
print(processed_str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_pathCE", type=str, default='./ckpt_AWGN_CE')
    parser.add_argument("--ckpt_pathRL", type=str, default='./ckpt_AWGN_RL')  # or './ckpt_AWGN_RL'
    args = parser.parse_args()

    # dict_train = pickle.load(open('./1.train_dict.pkl', 'rb'))
    dict_train = pickle.load(open('/home/eric/30_SemanticRL/test/SemanticRL/train_dict.pkl', 'rb'))
    rev_dict = {vv: kk for kk, vv in dict_train.items()}

    success_count = 0
    failure_count = 0
    with open('log.txt', 'w') as log_file:
        for input_str in SemanticRL_example:

            input_vector = [dict_train[x] for x in input_str.split(' ')] + [2]
            input_len = len(input_vector)
            input_vector = torch.tensor(input_vector)

      
    
            for ckpt_dir in [args.ckpt_pathCE]:#, args.ckpt_pathRL
                model_name = os.path.basename(ckpt_dir)
    
                encoder.load_state_dict(torch.load('/home/eric/30_SemanticRL/test/SemanticRL/ckpt_AWGN_CE/encoder_epoch99.pth', map_location='cpu'))
                decoder.load_state_dict(torch.load('/home/eric/30_SemanticRL/test/SemanticRL/ckpt_AWGN_CE/decoder_epoch99.pth', map_location='cpu'))
                embeds_shared.load_state_dict(torch.load('/home/eric/30_SemanticRL/test/SemanticRL/ckpt_AWGN_CE/embeds_shared_epoch99.pth',  map_location='cpu'))

                # encoder.load_state_dict(torch.load(ckpt_dir + '/encoder_epoch201.pth', map_location='cpu'))
                # decoder.load_state_dict(torch.load(ckpt_dir + '/decoder_epoch201.pth', map_location='cpu'))
                # embeds_shared.load_state_dict(torch.load(ckpt_dir + '/embeds_shared_epoch201.pth',  map_location='cpu'))
                # current_time = datetime.datetime.now()
                # print("Time:", current_time)
                for _ in range(20000):
                    output = do_test(input_vector.unsqueeze(0), encoder, decoder, normlize_layer, channel,
                        len_batch=torch.tensor(input_len).view(-1, ))
                    output = output.cpu().numpy()[0]
                    res = ' '.join(rev_dict[x] for x in output if x!=0 and x!=2)  # remove 'PAD' and 'EOS'

                    if res == processed_str:
                        success_count += 1
                    else:
                        failure_count += 1
                # current_time = datetime.datetime.now()
                # print("Time:", current_time)  
                    
                # print('result of {}:            {}'.format(model_name, res))
                    print('{}'.format(res))
                  

                      # Print the success and failure counts
    print('Success count:', success_count)
    print('Failure count:', failure_count)