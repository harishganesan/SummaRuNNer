import numpy as np
import torch


class Vocab():
    def __init__(self, flags, logger):
        word2id , self.embed = self.prepare_vocab_embeddingdict(flags, logger)

        self.id2word = {v:k for k,v in word2id.items()}
        assert len(word2id) == len(self.id2word)
    
    def __len__(self):
        return len(self.id2word)

    def i2w(self,idx):
        return self.id2word[idx]
    
    def prepare_vocab_embeddingdict(self, flags, logger):
        dtype = np.float32

        vocab_dict = {}
        word_embedding_array = []

        vocab_dict["_PAD"] = 0
        vocab_dict["_UNK"] = 1

        # Read word embedding file
        wordembed_filename = flags.embedding_file
        logger.info("Reading pretrained word embeddings file: %s" % wordembed_filename)

        embed_line = ""
        with open(wordembed_filename, "r") as fembedd:
            for linecount, line in enumerate(fembedd):
                if linecount == 0:
                    vocabsize = int(line.split()[0])
                    word_embedding_array = np.empty((vocabsize+2, flags.embed_dim), dtype)
                else:
                    linedata = line.split()
                    vocab_dict[linedata[0]] = linecount + 1
                    embeddata = [float(item) for item in linedata[1:]][0:flags.embed_dim]
                    word_embedding_array[linecount - 1] = embeddata

            logger.info("Done reading pre-trained word embedding of shape {}".format(word_embedding_array.shape))

        word_embedding_array[vocabsize:] = [0] * flags.embed_dim
        return vocab_dict, torch.from_numpy(word_embedding_array).float()