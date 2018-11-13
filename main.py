#!/usr/bin/env python3

import json
import models
from utils import BatchLoader
import argparse,random,logging,numpy,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from time import time
from tqdm import tqdm

parser = argparse.ArgumentParser()

# model
parser.add_argument('-save_dir',type=str,default='checkpoints/')
parser.add_argument('-embed_dim',type=int,default=100)
parser.add_argument('-embed_num',type=int,default=100)
parser.add_argument('-pos_dim',type=int,default=50)
parser.add_argument('-pos_num',type=int,default=100)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-hidden_size',type=int,default=200)
# train
parser.add_argument('-logfile', type=str)
parser.add_argument('-lr',type=float,default=1e-3)
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=5)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-train_dir',type=str,default='train_dir')
# parser.add_argument('-val_dir',type=str,default='data/val.json')
parser.add_argument('-report_every',type=int,default=1500)
parser.add_argument('-max_norm',type=float,default=1.0)
parser.add_argument('-use_trained', type=str,default='')
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1.pt')
parser.add_argument('-test_dir',type=str,default='data/test.json')
parser.add_argument('-ref',type=str,default='outputs/ref')
parser.add_argument('-hyp',type=str,default='outputs/hyp')
parser.add_argument('-topk',type=int,default=3)
# device
parser.add_argument('-device',type=int)
# option
parser.add_argument('-word2id',type=str,default='data/word2id.json')
parser.add_argument('-embedding',type=str,default='data/embedding.npz')
parser.add_argument('-max_doc_length',type=int,default=100)
parser.add_argument('-max_sent_length',type=int,default=50)
parser.add_argument('-target_label_size',type=int,default=2)
parser.add_argument('-num_sample_rollout',type=int,default=10)
parser.add_argument('-preprocessed_data_dir',type=str,default='data/preprocessed')
parser.add_argument('-data_mode',type=str,default='cnn')
parser.add_argument('-test',action='store_true')


args = parser.parse_args()
use_gpu = args.device is not None


if torch.cuda.is_available() and not use_gpu:
    print("WARNING: You have a CUDA device, should run with -device 0")

# set cuda device and seed
if use_gpu:
    torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed) 


def get_logger():
    logging.basicConfig(format='%(asctime)s,%(msecs)-2d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        filename=args.logfile,
                        level=logging.INFO)
    logger = logging.getLogger('SummaRuNNer-RNN-RL')
    return logger

# def eval(net,vocab,data_iter,criterion):
#     # function calculates aggregate loss on validation set
#     net.eval()
#     total_loss = 0
#     batch_num = 0
#     for batch in data_iter:
#         features,targets,_,doc_lens = vocab.make_features(batch)
#         features,targets = Variable(features), Variable(targets.float())
#         if use_gpu:
#             features = features.cuda()
#             targets = targets.cuda()
#         probs = net(features,doc_lens)
#         loss = criterion(probs,targets)
#         total_loss += loss.data.item()
#         batch_num += 1
#     loss = total_loss / batch_num
#     net.train()
#     return loss

def train():
    logger = get_logger()

    logger.info('Loading vocabulary dictionary and word embeddings...')

    # load word embeddings
    embed = torch.Tensor(np.load(args.embedding)['embedding'])

    # load word2id dictionary
    with open(args.word2id) as f:
        vocab_dict = json.load(f)

    logger.info('Loading training and validation datasets...')

    train_batcher = BatchLoader(args, vocab_dict, logger=logger, data_type='training')
    valid_batcher = BatchLoader(args, vocab_dict, logger=logger, data_type='validation')

    # update args
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)

    # instantiate model
    net = models.RNN_RNN(args, embed)
    if args.use_trained:
        logger.info('Training pre-trained model: %s' % args.use_trained)
        net.load_state_dict(torch.load(args.use_trained))
    if use_gpu:
        net.cuda()
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    trainer = net.model_wrapper(optimizer, mode='training')
    validator = net.model_wrapper(optimizer, mode='validation')

    # model info
    logger.info(net)
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    logger.info('Number of model parameters: %.1fM\n' % (params))
    
    min_loss = float('inf')
    t1 = time() 

    logger.info('Training started!\n')
    for epoch in range(args.epochs):

        batch_number = 0

        for batch in train_batcher.next_batch(args.batch_size):
            # features,targets,_,doc_lens = vocab_dict.make_features(batch)
            # features,targets = Variable(features), Variable(targets.float())
            # if use_gpu:
            #     features = features.cuda()
            #     targets = targets.cuda()

            loss, accuracy = trainer(batch)

            # make forward propogation
            probs = net(features,doc_lens)
            # calculate loss
            loss = criterion(probs,targets)
            # clear gradients
            optimizer.zero_grad()
            # back propogation
            loss.backward()
            # clip the gradient
            clip_grad_norm(net.parameters(), args.max_norm)
            # perform a single optimization step
            optimizer.step()

            # if args.debug:
            #     if logbatch:
            #         logbatch.write('{}:{}\n'.format(i, loss.data.item()))
            #     print('Batch ID:%d Loss:%f' %(i,loss.data.item()))
            # if i % args.report_every == 0:
            #     cur_loss = eval(net,vocab_dict,val_iter,criterion)
            #     if cur_loss < min_loss:
            #         min_loss = cur_loss
            #         best_path = net.save()
            #     if logepoch:
            #         logepoch.write('{}:{}:{}\n'.format(epoch, min_loss, cur_loss))
            #     logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f' % (epoch,min_loss,cur_loss))
            batch_number += 1

    t2 = time()
    logging.info('Total time:%f h'%((t2-t1)/3600))


def test():
    # load word embeddings
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    # load word2id dictionary
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    # load test dataset
    with open(args.test_dir) as f:
        examples = [json.loads(line) for line in f]
    test_dataset = utils.Dataset(examples)

    # instantiate batcher
    test_iter = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir, map_location='cuda:0')
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    # load the model and instantiate it
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    # load pretrained states
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    
    doc_num = len(test_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(test_iter):
        features,_,summaries,doc_lens = vocab.make_features(batch)
        t1 = time()
        # run the model over all the sentences of the batch
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        # probs: probabilities of all sentences of all the documents in the batch
        t2 = time()
        time_cost += t2 - t1
        start = 0
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len  # index of the last sentencse doc with doc_id
            # probabilities of sentences of doc with doc_id
            prob = probs[start:stop] if probs.dim() == 1 else torch.Tensor([probs])
            # how many top sentences to pick ?
            topk = min(args.topk,doc_len)
            # indices of k sentences with highest probabilities
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            # sort the indices
            topk_indices.sort()
            # get full doc splitted by sentences
            doc = batch['doc'][doc_id].split('\n')[:doc_len]
            # get topk sentences from the doc
            hyp = [doc[index] for index in topk_indices]
            # get golden summary for the doc
            ref = summaries[doc_id]
            # save machine and golden summary for the doc
            with open(os.path.join(args.ref,str(file_id)+'.txt'), 'w') as f:
                f.write(ref)
            with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w') as f:
                f.write('\n'.join(hyp))
            start = stop
            file_id = file_id + 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))


if __name__=='__main__':
    if args.test:
        test()
    else:
        train()
