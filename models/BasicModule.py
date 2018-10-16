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
