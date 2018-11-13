import torch
from torch.autograd import Variable
class BasicModule(torch.nn.Module):

    def __init__(self, args):
        super(BasicModule,self).__init__()
        self.args = args
        self.model_name = str(type(self))

    def pad_doc(self,words_out,doc_lens):
        pad_dim = words_out.size(1)
        # max number of sentences
        max_doc_len = max(doc_lens)
        sent_input = []
        start = 0
        for doc_len in doc_lens:
            # index of the last sentence in the doc
            stop = start + doc_len
            valid = words_out[start:stop]  # [docLength, 2*HiddenStates]
            start = stop
            if doc_len == max_doc_len:
                sent_input.append(valid.unsqueeze(0)) # [1, docMaxLength, 2*HiddenStates]
            else:
                pad = Variable(torch.zeros(max_doc_len-doc_len,pad_dim))
                if self.args.device is not None:
                    pad = pad.cuda()
                sent_input.append(torch.cat([valid,pad]).unsqueeze(0))
        sent_input = torch.cat(sent_input,dim=0) # [docs, docMaxLength, 2*HiddenStates]
        return sent_input

    def get_cross_entropy(self, logits, weights, oracle_multiple, reward_multiple):
        bce = torch.nn.BCELoss(reduction='none')

    def get_accuracy_metrics(self, logits, labels, weights):
        _, _, num_classes = logits.shape
        logits = logits.view(-1, num_classes)
        labels = labels.view(-1, num_classes)

        accuracy = 0

        for c in range(num_classes):
            true_labels = torch.argmax(labels, dim=1)
            true_labels_for_current_class = true_labels == c
            true_labels_for_current_class = true_labels_for_current_class.type(torch.cuda.FloatTensor) if self.args.device \
                else true_labels_for_current_class.type(torch.FloatTensor)

            predictions_for_class = torch.argmax(logits, dim=1) == c
            predictions_for_class = predictions_for_class.type(torch.cuda.FloatTensor) if self.args.device \
                else predictions_for_class.type(torch.FloatTensor)

            weights = weights.view(-1)
            weights = weights.type(torch.cuda.FloatTensor) if self.args.device else weights.type(torch.FloatTensor)

            effective_number_of_samples = weights.sum()  # 4

            # true_labels_for_current_class = torch.mul(true_labels_for_current_class, weights)  # [1, 1, 0, 1, 0]
            # predictions_for_class = torch.mul(predictions_for_class, weights)  # [1, 1, 0, 1, 0]

            true_positives_for_class = true_labels_for_current_class * predictions_for_class
            accuracy += true_positives_for_class.sum().float()

        accuracy /= effective_number_of_samples

        return accuracy

    def model_wrapper(self, optimizer, mode='training'):
        def _wrapper(data):
            data = [Variable(torch.from_numpy(var)) for var in data]
            data = [data[0].long(), data[1].long(), data[2].long(), data[3].long(), data[4].float()]
            data = [var.cuda() if self.args.device else var for var in data]
            [batch_docs, batch_docs_length, batch_targets, batch_oracle_targets, batch_rewards] = data

            # total_num_sentences = batch_weights.view(-1)
            # true_labels = true_labels.type(torch.cuda.FloatTensor) if self.args.device else true_labels.type(torch.FloatTensor)
            total_num_sentences = batch_docs_length.sum()

            logits = self(batch_docs, batch_docs_length)

            cross_entropy = self.get_cross_entropy(logits, batch_docs_length, batch_oracle_targets, batch_rewards)
            loss = cross_entropy.mean()  # Mean should take acount the weights too! This mean should happen in get_cross_entropy
            accuracy = self.get_accuracy_metrics(logits, batch_targets, batch_docs_length)

            if mode == "training":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            return loss, accuracy, total_num_sentences

        if mode == 'validation':
            def _validator(batcher):
                batcher.shuffle()
                total_loss = 0.0
                total_accuracy = 0.0
                total_num_sentences = 0
                self.eval()
                for data in batcher.next_batch(self.args.batch_size):
                    loss, accuracy, batch_sentence_num = _wrapper(data)
                    total_loss += (loss * batch_sentence_num)
                    total_accuracy += (accuracy * batch_sentence_num)
                    total_num_sentences += batch_sentence_num
                total_loss = total_loss / total_num_sentences
                total_accuracy = total_accuracy / total_num_sentences
                self.train()
                return total_loss, total_accuracy

            return _validator

        return _wrapper

    def save(self):
        checkpoint = {'model':self.state_dict(), 'args': self.args}
        best_path = '%s%s_seed_%d.pt' % (self.args.save_dir,self.model_name,self.args.seed)
        torch.save(checkpoint,best_path)

        return best_path

    def load(self, best_path):
        if self.args.device is not None:
            data = torch.load(best_path)['model']
        else:
            data = torch.load(best_path, map_location=lambda storage, loc: storage)['model']
        self.load_state_dict(data)
        if self.args.device is not None:
            return self.cuda()
        else:
            return self
