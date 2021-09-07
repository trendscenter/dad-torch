import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchtext.datasets import WikiText2

from dad_torch import DADTorch, NNTrainer, ConfusionMatrix, default_ap
from dad_torch.config import boolean_string
import argparse
from models.transformer import CTransformer, GTransformer
#from data.torchtext import ImdbDataset

ap = argparse.ArgumentParser(parents=[default_ap], add_help=False)
ap.add_argument('--ignore-backward', default=False, type=boolean_string, help='Ignore .backward in runtime record.')

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
])

from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import Dataset
from torchtext import data, datasets, vocab

import torch
import numpy as np

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset

tokenizer = get_tokenizer('basic_english')
train_iter = datasets.WikiText2(split='train')

tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x == 'pos')
bptt = 35
def get_batch(source, i: int):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

class MNISTTrainer(NNTrainer):
    def _init_nn_model(self):
        self.nn['model'] = GTransformer(128,
            8,
            1,
            256,
            len(vocab))

    def iteration(self, i, batch):
        ii = batch
        inputs, labels = get_batch(train_data, ii)
        inputs = inputs.t().to(self.device["gpu"])
        labels = labels.t().to(self.device["gpu"])

        #labels = batch[0].to(self.device["gpu"]).long()
        #inputs = batch[1].to(self.device["gpu"]).long()
        #inputs = inputs.reshape()
        #try:
        #    inputs = torch.flatten(batch[0].to(self.device['gpu']).float(), 1)
        #except IndexError:
        #    inputs = torch.flatten(batch[0].to(self.device['gpu']).float())
        #labels = batch[1].to(self.device['gpu']).long()
        # print('***********: ', self.device['gpu'], torch.initial_seed())

        out = self.nn['model'](inputs)
        out = out.view(-1, len(vocab))
        loss = F.nll_loss(out, labels)

        _, pred = torch.max(out, 1)
        sc = self.new_metrics()
        #sc.add(pred, labels.float())

        avg = self.new_averages()
        avg.add(loss.item(), len(inputs))

        return {'loss': loss, 'averages': avg, 'metrics': sc, 'predictions': pred}

    def init_experiment_cache(self):
        self.cache['log_header'] = 'Loss|Accuracy,F1,Precision,Recall'
        self.cache.update(monitor_metric='f1', metric_direction='maximize')

    def new_metrics(self):
        return ConfusionMatrix(num_classes=10)


#train_dataset = datasets.MNIST('data', train=True, download=True,
#                               transform=transform)
#val_dataset = datasets.MNIST('data', train=False,
#                             transform=transform)
#train_dataset = MyImdbDataset(train=True)
#val_dataset = MyImdbDataset(train=False)
iter = 64 * 1000
#train_dataset.data = 
#train_dataset.target = [train_dataset[i][1] for i in range(len(train_dataset))]

#val_dataset.data = [val_dataset[i][0] for i in range(len(val_dataset))]
#val_dataset.target = [val_dataset[i][1] for i in range(len(val_dataset))]


def data_process(raw_text_iter: dataset.IterableDataset):
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

def batchify(data, bsz: int):
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data

batch_size = 32
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

train_range = range(0, train_data.size(0) - 1, bptt)
test_range = range(0, test_data.size(0) - 1, bptt)

dataloader_args = {'train': {'dataset': train_range, "batch_size": 1, "shuffle": False, "drop_last": True},
                   'test': {'dataset': test_range,  "batch_size": 1, "shuffle":False, "drop_last": True}}

if __name__ == "__main__":
    runner = DADTorch(dataloader_args=dataloader_args, args=ap, seed=3, seed_all=True, force=True)
    runner.run(MNISTTrainer)
