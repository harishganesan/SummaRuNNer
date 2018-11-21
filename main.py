#!/usr/bin/env python3

import json
import models
from utils import BatchLoader, Vocab
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
parser.add_argument('-logfile', type=str,default='logfile.log')
parser.add_argument('-lr',type=float,default=1e-3)
parser.add_argument('-batch_size',type=int,default=20)
parser.add_argument('-epochs',type=int,default=5)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-val_per_epoch',type=int,default=8)
parser.add_argument('-max_norm',type=float,default=1.0)
parser.add_argument('-use_trained', type=str,default='')
# test
parser.add_argument('-test_model',type=str,default='checkpoints/RNN_RL_seed_1.pt')
parser.add_argument('-ref',type=str,default='outputs/ref')
parser.add_argument('-hyp',type=str,default='outputs/hyp')
# device
parser.add_argument('-device',type=int)
# option
parser.add_argument('-embedding_file',type=str,default='data/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec')
parser.add_argument('-max_doc_length',type=int,default=100)
parser.add_argument('-max_sent_length',type=int,default=50)
parser.add_argument('-num_sample_rollout',type=int,default=10)
parser.add_argument('-preprocessed_data_dir',type=str,default='data/preprocessed')
parser.add_argument('-data_mode',type=str,default='cnn-dailymail')
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


def get_logger(logfile=None):
    formatter = logging.Formatter('%(asctime)s.%(msecs)-2d %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                                  datefmt='%d-%m-%Y %H:%M:%S')
    logger = logging.getLogger('SummaRuNNer-RL')
    logger.setLevel(logging.INFO)
    if logfile or args.logfile:
        fh = logging.FileHandler(logfile or args.logfile, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def train():
    logger = get_logger()

    # load word embeddings
    vocab = Vocab(args, logger)

    # load datasets
    train_batcher = BatchLoader(args, logger=logger, data_type='training')
    valid_batcher = BatchLoader(args, logger=logger, data_type='validation')

    # update args
    args.embed_num = vocab.embed.size(0)
    args.embed_dim = vocab.embed.size(1)

    # instantiate model
    net = models.RNN_RNN(args, vocab.embed)
    if args.use_trained:
        logger.info('Training pre-trained model: %s' % args.use_trained)
        net.load(args.use_trained)
    if args.device is not None:
        net.cuda()
    net.train()


    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    trainer = net.model_wrapper(optimizer, mode='training')
    validator = net.model_wrapper(optimizer, mode='validation')

    # model info
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    logger.info('Number of model parameters: %.1fM\n' % (params))
    
    min_loss = float('inf')
    total_batches = numpy.ceil(train_batcher.get_data_size() / args.batch_size)
    validate_every = numpy.ceil(total_batches / args.val_per_epoch)
    t1 = time() 

    logger.info('Batches per epoch = %d', total_batches)
    logger.info('Set to validate every %d batches', validate_every)
    logger.info('Training started!\n')
    for epoch in range(args.epochs):

        batch_number = 0
        train_batcher.shuffle()

        for batch in train_batcher.next_batch(args.batch_size):
            abs_batch = epoch * total_batches + batch_number
            docs_processed = batch_number * args.batch_size

            model_file = '%s_%d_%d_%d_seed_%d' % (
                net.model_name, epoch, batch_number, abs_batch, args.seed)

            # checkpoints

            if epoch == 0 and batch_number == 0:
                logger.info('Checkpoint: saving model to file %s', model_file)
                net.save(filename=model_file)

            if epoch == 0 and batch_number == (total_batches // 4):
                logger.info('Checkpoint: saving model to file %s', model_file)
                net.save(filename=model_file)

            if epoch == 1 and batch_number == 0:
                logger.info('Checkpoint: saving model to file %s', model_file)
                net.save(filename=model_file)

            if epoch == 2 and batch_number == 0:
                logger.info('Checkpoint: saving model to file %s', model_file)
                net.save(filename=model_file)

            if batch_number % validate_every == 0:
                # validate
                logger.info('Validating model at batch %d...', abs_batch)
                v_loss, v_accuracy = validator(valid_batcher)
                v_loss *= 1e3
                logger.info('%d/%d/%d (Ep/Batch/AbBatch): Val loss = %.2f, '
                            'Val accuracy = %.4f', epoch, batch_number, abs_batch,
                            v_loss, v_accuracy)
                if v_loss < min_loss:
                    min_loss = v_loss
                    logger.info('New loss %.2f < %.2f, saving the model...',
                                 v_loss, min_loss)
                    path = net.save()
                    logger.info(' model saved in %s', path)
                else:
                    logger.info('New loss %.2f > %.2f, keep going...',
                                v_loss, min_loss)

            logits, loss, accuracy = trainer(batch)

            logger.info('%d/%d/%d (Ep/Batch/AbBatch): Processed = %d, '
                        'Loss = %.2f, Accuracy = %.4f', epoch, batch_number,
                        abs_batch, docs_processed, loss*1e3, accuracy)

            batch_number += 1

    t2 = time()
    logging.info('Total time: %f h' % ((t2-t1)/3600))


def test():
    logger = get_logger('testlog.log')

    # load word embeddings
    vocab = Vocab(args, logger)

    test_batcher = BatchLoader(args, logger=logger, data_type='validation') # TODO change to test

    # load model
    logger.info('Loading model to test: %s' % args.test_model)
    if use_gpu:
        checkpoint = torch.load(args.test_model, map_location='cuda:0')
    else:
        checkpoint = torch.load(args.test_model, map_location=lambda storage, loc: storage)
        checkpoint['args'].device = None
    net = models.RNN_RNN(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()

    tester = net.model_wrapper(None, mode='test')

    file_id = 0
    for batch in tqdm(test_batcher.next_batch(args.batch_size)):
        logits, _, _ = tester(batch)
        batch_docs, batch_docs_length, batch_targets, *_ = batch

        start = 0
        for doc_id, doc_len in enumerate(batch_docs_length):
            stop = start + doc_len  # index of the last sentencse doc with doc_id
            sentences = batch_docs[start:stop]
            # indices of targets
            doc_targets = np.where(batch_targets[start:stop] == 1)[0]
            # how many top sentences to pick ?
            topk = len(doc_targets)
            # probabilities of sentences of doc with doc_id
            doc_probs = logits[start:stop]
            # indices of k sentences with highest probabilities
            topk_indices = doc_probs.topk(topk)[1].cpu().data.numpy()
            # sort the indices
            topk_indices.sort()

            ref = '\n'.join([
                ' '.join(map(vocab.i2w, np.trim_zeros(sentences[t]))) for t in doc_targets
            ])
            hyp = '\n'.join([
                ' '.join(map(vocab.i2w, np.trim_zeros(sentences[t]))) for t in topk_indices
            ])

            with open(os.path.join(args.ref, str(file_id) + '.txt'), 'w') as f:
                f.write(ref)
            with open(os.path.join(args.hyp, str(file_id) + '.txt'), 'w') as f:
                f.write(hyp)

            start = stop
            file_id += 1
    logger.info('Done!')


if __name__=='__main__':
    if args.test:
        test()
    else:
        train()
