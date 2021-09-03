import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchtext.datasets import IMDB

from dad_torch import DADTorch, NNTrainer, ConfusionMatrix, default_ap
from dad_torch.config import boolean_string
import argparse
from models.transformer import CTransformer
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

tokenizer = get_tokenizer('basic_english')
train_iter = datasets.IMDB(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x == 'pos')

class MNISTTrainer(NNTrainer):
    def _init_nn_model(self):
        self.nn['model'] = CTransformer(128,
            8,
            1,
            256,
            len(vocab),
            2,
            False)

    def iteration(self, batch):
        labels = batch[0].to(self.device["gpu"]).long()
        inputs = batch[1].to(self.device["gpu"]).long()
        #inputs = inputs.reshape()
        #try:
        #    inputs = torch.flatten(batch[0].to(self.device['gpu']).float(), 1)
        #except IndexError:
        #    inputs = torch.flatten(batch[0].to(self.device['gpu']).float())
        #labels = batch[1].to(self.device['gpu']).long()
        # print('***********: ', self.device['gpu'], torch.initial_seed())

        out = self.nn['model'](inputs)
        loss = F.nll_loss(out, labels)

        _, pred = torch.max(out, 1)
        sc = self.new_metrics()
        sc.add(pred, labels.float())

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

train_dataset=to_map_style_dataset(IMDB(split="train"))
val_dataset=to_map_style_dataset(IMDB(split="test"))

def collate_batch(batch, max_len=256):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:         
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         while len(processed_text) < max_len:
             processed_text = torch.cat([processed_text, torch.zeros(1,)])        
         text_list.append(processed_text[:max_len])
         label_list.append(label_pipeline(_label))
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.stack(text_list, 0)
    return label_list, text_list, offsets

dataloader_args = {'train': {'dataset': train_dataset, 'collate_fn': collate_batch, "batch_size": 64},
                   'test': {'dataset': val_dataset, 'collate_fn': collate_batch, "batch_size": 64}}

if __name__ == "__main__":
    runner = DADTorch(dataloader_args=dataloader_args, args=ap, seed=3, seed_all=True, force=True)
    runner.run(MNISTTrainer)
