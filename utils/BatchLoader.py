import numpy as np
import random


class Data:
    def __init__(self, flags, vocab_dict, data_type, logger):
        self.PAD_ID = 0

        self.flags = flags
        self.logger = logger
        self.filenames = []
        self.docs = []
        self.labels = []
        self.weights = []
        self.rewards = []
        self.fileindices = []

        self.data_type = data_type

        # populate the data
        self.populate_data(vocab_dict, data_type)

    def __len__(self):
        return len(self.fileindices)

    def shuffle_fileindices(self):
        random.shuffle(self.fileindices)

    def get_batch(self, startidx, endidx):
        dtype = np.float32

        def process_to_chop_pads(ordids, requiredsize):
            if len(ordids) >= requiredsize:
                return ordids[:requiredsize]
            else:
                padids = [self.PAD_ID] * (requiredsize - len(ordids))
                return ordids + padids

        # For training, (endidx-startidx)=FLAG.batch_size
        # For others, it is as specified
        batch_docnames = np.empty((endidx-startidx), dtype="S60")  # file id

        batch_docs = []
        batch_docs_length = np.empty(endidx-startidx, dtype='int32')
        batch_targets = []
        batch_oracle_targets = []
        batch_rewards = np.empty(endidx - startidx, dtype=dtype)

        for batch_idx, fileindex in enumerate(self.fileindices[startidx:endidx]):
            # Doc name
            batch_docnames[batch_idx] = self.filenames[fileindex]

            # Doc padded sentences
            for idx in range(len(self.docs[fileindex])):
                sent = self.docs[fileindex][idx][:]
                sent = process_to_chop_pads(sent, self.flags.max_sent_length)
                batch_docs.append(sent)

            # Doc length
            batch_docs_length[batch_idx] = sum(self.weights[fileindex])

            # Target
            batch_targets.extend([1 if (item in self.labels[fileindex][0]) else 0 for item in range(batch_docs_length[batch_idx])])

            # Oracle target and reward
            random_oracle = random.randint(0, (self.flags.num_sample_rollout - 1))
            if random_oracle < len(self.labels[fileindex]):
                oracle_labels = [1 if (item in self.labels[fileindex][random_oracle]) else 0 for item in range(batch_docs_length[batch_idx])]
                batch_rewards[batch_idx] = self.rewards[fileindex][random_oracle]
            else:
                # pick the best oracle
                oracle_labels = [1 if (item in self.labels[fileindex][0]) else 0 for item in range(batch_docs_length[batch_idx])]
                # pick best reward
                batch_rewards[batch_idx] = self.rewards[fileindex][0]
            batch_oracle_targets.extend(oracle_labels)

        batch_docs = np.array(batch_docs, dtype='int32')
        batch_targets = np.array(batch_targets, dtype='int32')
        batch_oracle_targets = np.array(batch_oracle_targets, dtype='int32')

        return batch_docs, batch_docs_length, batch_targets, batch_oracle_targets, batch_rewards

    def populate_data(self, vocab_dict, data_type):

        full_data_file_prefix = self.flags.preprocessed_data_dir + "/" + self.flags.data_mode + "." + data_type

        # Process doc, title, image and label
        doc_data_list = open(full_data_file_prefix+".doc").read().strip().split("\n\n")
        label_data_list = open(full_data_file_prefix + ".label.multipleoracle").read().strip().split("\n\n")

        doccount = 0
        for doc_data, label_data in zip(doc_data_list, label_data_list):

            doc_lines = doc_data.strip().split("\n")
            label_lines = label_data.strip().split("\n")

            filename = doc_lines[0].strip()
            if filename == label_lines[0].strip():
                self.filenames.append(filename)

                # Doc
                thisdoc = []
                for line in doc_lines[1:self.flags.max_doc_length+1]:
                    thissent = [int(item) for item in line.strip().split()]
                    thisdoc.append(thissent)
                self.docs.append(thisdoc)

                # Weights
                originaldoclen = int(label_lines[1].strip())
                thisweight = [1 for item in range(originaldoclen)][:self.flags.max_doc_length]
                self.weights.append(thisweight)

                # Labels (multiple oracles and preestimated rewards)
                thislabel = [] # list of sentences (multiple oracles) [[20, 1, 4], [1, 45], ...]
                thisreward = [] # ROUGE for each oracle [0.47, 0.46, ...]
                for line in label_lines[2:self.flags.num_sample_rollout + 2]:
                    thislabel.append([int(item) for item in line.split()[:-1]])
                    thisreward.append(float(line.split()[-1]))
                self.labels.append(thislabel)
                self.rewards.append(thisreward)

            else:
                self.logger.error("Some problem with %s.* files. Exiting!" % full_data_file_prefix)
                exit(1)

            if doccount % 10000 == 0:
                self.logger.info("%d ..." % doccount)

            doccount += 1

        # Set file indices
        self.fileindices = list(range(len(self.filenames)))
        self.logger.info("Read {} docs".format(len(self.filenames)))


def prepare_news_data(flags, vocab_dict, logger, data_type="training"):
    data = Data(flags, vocab_dict, data_type, logger=logger)
    return data


class BatchLoader:
    def __init__(self, flags, vocab_dict, logger, data_type="training"):
        self.data = prepare_news_data(flags, vocab_dict, logger, data_type)

    def shuffle(self):
        self.data.shuffle_fileindices()

    def get_all_data(self):
        return self.data.get_batch(startidx=0, endidx=len(self.data))

    def get_batch(self, startidx, endidx):
        return self.data.get_batch(startidx, endidx)

    def next_batch(self, batch_size):
        for startidx in range(0, len(self.data), batch_size):
            #batch_docs, batch_label, batch_weights, batch_oracle_multiple, batch_reward_multiple = self.data.get_batch(startidx, startidx+batch_size)
            #yield (batch_docs, batch_label, batch_weights, batch_oracle_multiple, batch_reward_multiple)
            #yield self.data[startidx: startidx + batch_size]
            yield self.data.get_batch(startidx, startidx + batch_size)

if __name__ == "__main__":

    print("main pf data_utils")
    b = BatchLoader([0], "validation")
    print(len(b.get_all_data()[0]))