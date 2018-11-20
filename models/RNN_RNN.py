from .BasicModule import BasicModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNN_RNN(BasicModule):
    def __init__(self, args, embed=None):
        super(RNN_RNN, self).__init__(args)
        self.model_name = 'RNN_RL'
        self.args = args
        
        V = args.embed_num  # number of word embeddings
        D = args.embed_dim  # dimensionality of the embeddings
        H = args.hidden_size  # size of the hidden layers
        S = args.seg_num  # number of segments for relative position
        P_V = args.pos_num  # number of embeddings for absolute position
        P_D = args.pos_dim  # number of embeddings for relative position
        self.abs_pos_embed = nn.Embedding(P_V,P_D)
        self.rel_pos_embed = nn.Embedding(S,P_D)
        self.embed = nn.Embedding(V,D,padding_idx=0)  # word embeddings
        if embed is not None:
            self.embed.weight.data.copy_(embed)  # copy word embeddings

        self.word_RNN = nn.GRU(
                        input_size = D,
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
        self.sent_RNN = nn.GRU(
                        input_size = 2*H,
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
        self.fc = nn.Linear(2*H,2*H)

        # Parameters of Classification Layer
        self.content = nn.Linear(2*H,1,bias=False)
        self.salience = nn.Bilinear(2*H,2*H,1,bias=False)
        self.novelty = nn.Bilinear(2*H,2*H,1,bias=False)
        self.abs_pos = nn.Linear(P_D,1,bias=False)
        self.rel_pos = nn.Linear(P_D,1,bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1))

    def max_pool1d(self,x,seq_lens):
        # x:[sentences, wordsPadded, 2*HiddenStates]
        out = []
        for index,t in enumerate(x):
            # t: [wordsPadded, 2*HiddenStates]
            t = t[:seq_lens[index],:]  # t: [wordsPadded, 2*HiddenStates]
            t = torch.t(t).unsqueeze(0)  # t: [1, 2*HiddenStates, words]
            out.append(F.max_pool1d(t,t.size(2)))  # [1, 2*HiddenStates, 1]
        
        out = torch.cat(out).squeeze(2)  # out: [sentences, 2*HiddenStates]
        return out

    def forward(self,x,doc_lens):
        # x is two dimensional, [sentences, wordid]
        # calculate number of words in each sentence
        sent_lens = torch.sum(torch.sign(x),dim=1).data  # [39, 23, ...]
        # convert word ids into features
        x = self.embed(x) # [sentences, words, features]
        H = self.args.hidden_size  # 200

        # word level GRU
        x = self.word_RNN(x)[0]  # [sentences, words, 2*HiddenStates]
        word_out = self.max_pool1d(x,sent_lens)  # [sentences, 2*HiddenStates]
        
        # pad docs to make the same number of sentences
        x = self.pad_doc(word_out,doc_lens)  # x: [docs, MaxSentences, 2*HiddentStates]

        # sentence level GRU
        sent_out = self.sent_RNN(x)[0]  # [docs, MaxSentences, 2*HiddenStates]
        docs = self.max_pool1d(sent_out,doc_lens) # [docs, 2*HiddenStates]

        probs = []
        for index,doc_len in enumerate(doc_lens):
            # get hidden states for sentences of the document
            valid_hidden = sent_out[index,:doc_len,:]  # [sentences, 2*HiddenStates]
            doc = torch.tanh(self.fc(docs[index])).unsqueeze(0)  # [1, 2*HiddenStates]
            s = Variable(torch.zeros(1,2*H))
            if self.args.device is not None:
                s = s.cuda()
            for position, h in enumerate(valid_hidden):
                # position - index of a sentence within the current document
                # h - hidden states from the sentence layer for current sentence

                # reshape to tensor of tensors
                h = h.view(1, -1) # [1, 2*HiddenStates]

                abs_index = Variable(torch.LongTensor([[position]]))
                if self.args.device is not None:
                    abs_index = abs_index.cuda()
                # convert absolute position of the sentence in the doc into feature vector
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)  # [1, 50]
                
                # relative position of the sentence within segment [0, 10)
                rel_index = int(round((position + 1) * 9.0 / int(doc_len)))
                rel_index = Variable(torch.LongTensor([[rel_index]]))
                if self.args.device is not None:
                    rel_index = rel_index.cuda()
                # convert relative position of the sentence in the segment into feature vector
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)
                
                # classification layer
                content = self.content(h) 
                salience = self.salience(h,doc)
                novelty = -1 * self.novelty(h, torch.tanh(s))
                abs_p = self.abs_pos(abs_features)
                rel_p = self.rel_pos(rel_features)
                prob = torch.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
                s = s + torch.mm(prob,h)
                probs.append(prob)
        return torch.cat(probs).squeeze()  # return sentence probabilities for all documents
