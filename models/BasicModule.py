import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

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
                if self.args.device is not None:
                    valid = valid.cuda()
                sent_input.append(valid.unsqueeze(0)) # [1, docMaxLength, 2*HiddenStates]
            else:
                pad = Variable(torch.zeros(max_doc_len-doc_len,pad_dim))
                if self.args.device is not None:
                    pad = pad.cuda()
                    valid = valid.cuda()
                sent_input.append(torch.cat([valid,pad]).unsqueeze(0))
        sent_input = torch.cat(sent_input, dim=0) # [docs, docMaxLength, 2*HiddenStates]
        return sent_input

    def get_cross_entropy(self, logits, docs_length, oracle_targets, rewards):
        bce = torch.nn.BCELoss(reduction='none')
        entropy = bce(logits, oracle_targets)

        reward_tensor = torch.Tensor([])
        if self.args.device is not None:
            reward_tensor = reward_tensor.cuda()
        for i, reward in enumerate(rewards):
            reward_tensor = torch.cat((reward_tensor, reward.repeat(int(docs_length[i]))))

        entropy = torch.mul(entropy, reward_tensor)

        return entropy.mean()

    def get_accuracy_metrics(self, logits, targets, docs_length):
        accuracy = 0

        accuracy += torch.mul(targets == 0, logits < 0.5).sum().float()
        accuracy += torch.mul(targets == 1, logits >= 0.75).sum().float()

        accuracy /= len(targets)

        return accuracy

    def model_wrapper(self, optimizer, mode='training'):
        def _wrapper(data):
            data = [torch.from_numpy(data[i]) for i in range(len(data))]
            if self.args.device is not None:
                data = [data[i].cuda() for i in range(len(data))]
            [batch_docs, batch_docs_length, batch_targets, batch_oracle_targets, batch_rewards] = data

            logits = self(batch_docs, batch_docs_length)

            loss = self.get_cross_entropy(logits, batch_docs_length, batch_oracle_targets, batch_rewards)

            accuracy = self.get_accuracy_metrics(logits, batch_targets, batch_docs_length)

            if mode == "training":
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.parameters(), self.args.max_norm)
                optimizer.step()

            return logits, loss, accuracy

        if mode == 'validation':
            def _validator(batcher):
                batcher.shuffle()
                total_loss = 0.0
                total_accuracy = 0.0
                total_batches = 0
                self.eval()
                for data in batcher.next_batch(self.args.batch_size):
                    _, loss, accuracy = _wrapper(data)
                    total_loss += float(loss)
                    total_accuracy += float(accuracy)
                    total_batches += 1
                total_loss = total_loss / max(1, total_batches)
                total_accuracy = total_accuracy / max(1, total_batches)
                self.train()
                return total_loss, total_accuracy

            return _validator

        return _wrapper

    def save(self, filename=None):
        checkpoint = {'model':self.state_dict(), 'args': self.args}
        if not filename:
            filename = '%s_seed_%d' % (self.model_name, self.args.seed)
        best_path = '%s%s.pt' % (self.args.save_dir, filename)
        torch.save(checkpoint, best_path)

        return best_path

    def load(self, best_path):
        if self.args.device is not None:
            data = torch.load(best_path, map_location='cuda:0')['model']
        else:
            data = torch.load(best_path, map_location=lambda storage, loc: storage)['model']
        self.load_state_dict(data)
        if self.args.device is not None:
            return self.cuda()
        else:
            return self
