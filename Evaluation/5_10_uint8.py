"""
This work is created by KunLu. Copyright reserved.
lukun199@gmail.com
19th Feb., 2021

# Inference.py
"""
import os, platform, json, time, pickle, sys, argparse
import torch
from math import log
sys.path.append('./')
from data_loader import Dataset_sentence_test, collate_func
from model import LSTMEncoder, LSTMDecoder, Embeds
from utils import Normlize_tx, Channel, smaple_n_times

import logging
import datetime
import random
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction

_snr = 10
_iscomplex = True
channel_dim = 10
n_bit = 8

NUM_BITS = 8
def fixed_point2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, dtype=torch.uint8)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2fixed_point(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, dtype=torch.uint8)
    return torch.sum(mask * b, -1).to(torch.uint8)



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


device = torch.device("cpu:0")
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
embeds_shared = Embeds(vocab_size=300, num_hidden=512).to(device)
encoder = LSTMEncoder(channel_dim=channel_dim, embedds=embeds_shared).to(device)
decoder = LSTMDecoder(channel_dim=channel_dim, embedds=embeds_shared, vocab_size = 300).to(device)

encoder = encoder.eval()
decoder = decoder.eval()
embeds_shared = embeds_shared.eval()


normlize_layer = Normlize_tx(_iscomplex=_iscomplex)
channel = Channel(_iscomplex=_iscomplex)

def do_test(input_data, encoder, decoder, normlize_layer, channel, len_batch):

    with torch.no_grad():
        print("input_data :")
        print(input_data)
        output, _ = encoder(input_data, len_batch)
        print("encoder :")
        print(output)
        output = normlize_layer.apply(output)
        print("normlize_layer :")
        print(output)
        


        range_xf = torch.max(output) - torch.min(output)
#print('range:{}'.format(range_xf))
        alpha = (2**(n_bit-0)-1)/(range_xf)
#print('alpha:{}'.format(alpha))
        zp = torch.round(torch.min(output) * alpha)
#print('zeropoint:{}'.format(zp))
        pred1 = torch.round(alpha*output) - zp
        print("float_point convert fixed_point :")
#print(pred1)

        pred1 = torch.floor(pred1)
        pred1 = pred1.long()

        print(pred1)
        e = 7

       
        fixed_point = fixed_point2bin(pred1, NUM_BITS)
        print("fixed_point convert bit vector :")
        pred1 = fixed_point.squeeze(0)
        print(pred1)

        pred1 = flip_bits(pred1,2)
        
        print("channel noise :")
        print(pred1)

        
        print("bit vector convert fixed_point :")
        pred1 = bin2fixed_point(pred1, NUM_BITS)
        pred2 = pred1.unsqueeze(0)
        print(pred2)

        e = 7
        print("fixed_point convert float_point :")

        pred2 = pred2 / 1.0


        pred8 = (pred2+zp) / alpha


        #pred8 = torch.where(output < 0, -de_xf, de_xf)

        print(pred8)  

        output = decoder.sample_max_batch(pred8, None)
        print("decoder :")
        print(output)

    return output

SemanticRL_example = [          
                                
                     
                 #       'downlink information and vrb will map to prb and retransmission and a serving cell received',
                         'planets engage in an eternal graceful dance around sun rise'
                       #  'vrb directly mapped to prb',
                        #  'from ue to gnb'
                      ]
#input_str = "['this message will send downlink information'\n'vrb to prb is an interleaved mapping way' ]"
input_str = "['planets engage in an eternal graceful dance around sun rise']"
#input_str = "['downlink information and vrb will map to prb and retransmission and a serving cell received']"


processed_str = input_str.strip("[]").replace("'", "")
print('--------------Reference sentence---------------')
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
                for _ in range(5000):
                    output = do_test(input_vector.unsqueeze(0), encoder, decoder, normlize_layer, channel,
                        len_batch=torch.tensor(input_len).view(-1, ))
                    output = output.cpu().numpy()[0]
                    res = ' '.join(rev_dict[x] for x in output if x!=0 and x!=2)  # remove 'PAD' and 'EOS'

                    # if res == processed_str:
                    #     success_count += 1
                    # else:
                    #     failure_count += 1
                # current_time = datetime.datetime.now()
                # print("Time:", current_time)  
                    
                # print('result of {}:            {}'.format(model_name, res))
                    print('--------------Candidate sentence---------------')
                    print('{}'.format(res))
                  

                    print('-----------------------------------------------')
                    print('------------------Comparison-------------------')
                    print('-----------------------------------------------')

                    sent_a_reference = 'planets engage in an eternal graceful dance around sun rise'.split()
                    print('sent_a_reference_sentence = {} '.format(sent_a_reference))
                    
                    sent_b_reference = 'moon casts gentle soft glow across dark darkness took green'.split()
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