import torch

class Vocab():
    def __init__(self,embed,word2id):
        self.embed = embed
        self.word2id = word2id
        self.id2word = {v:k for k,v in word2id.items()}
        assert len(self.word2id) == len(self.id2word)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'
    
    def __len__(self):
        return len(word2id)

    def i2w(self,idx):
        return self.id2word[idx]
    def w2i(self,w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX
    
    def make_features(self,batch,sent_trunc=50,doc_trunc=100,split_token='\n'):
        sents_list,targets,doc_lens = [],[],[]
        # trunc document
        for doc,label in zip(batch['doc'],batch['labels']):
            # split each document into list of sentences
            sents = doc.split(split_token)
            # get list of labels
            labels = label.split(split_token)
            # convert each label to integer
            labels = [int(l) for l in labels]
            # calculate max number of sentences, <= 100
            max_sent_num = min(doc_trunc,len(sents))
            # truncate number of sentences in the current doc
            sents = sents[:max_sent_num]
            # truncate the corresponding labels list
            labels = labels[:max_sent_num]
            # add sentences of the doc to sentence list
            sents_list += sents
            # add the labels to label list
            targets += labels
            doc_lens.append(len(sents))
        # trunc or pad sentence
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:  # sents_list: ['doc0sent0', 'doc0sent1', 'doc1sent0', ...]
            # split each of the sentences into words
            words = sent.split()
            if len(words) > sent_trunc:
                # truncate number of words in sentence, <= 50
                words = words[:sent_trunc]
            # remember max number of words in a sentence
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            # add splitted sentence to batch list
            batch_sents.append(words)  # [[d0s0w0, d0s0w1], [d0s1w0, d0s1w1], ...]
        
        features = []
        for sent in batch_sents:
            # convert each word in a sentence into integer representation
            # pad each sentence with 0 to reach max sentence length
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
            features.append(feature)
        
        features = torch.LongTensor(features)    
        targets = torch.LongTensor(targets)
        summaries = batch['summaries']
        # doc_lens - list of number of sentences in each document, [12, 21, ...]
        return features,targets,summaries,doc_lens
