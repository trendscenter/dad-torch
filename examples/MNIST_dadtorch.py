import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

from dad_torch import DADTorch, NNTrainer, ConfusionMatrix, default_ap
from dad_torch.config import boolean_string

ap = argparse.ArgumentParser(parents=[default_ap], add_help=False)
ap.add_argument('--ignore-backward', default=False, type=boolean_string, help='Ignore .backward in runtime record.')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# **Define neural network. I just burrowed from here: https://github.com/pytorch/examples/blob/master/mnist/main.py**
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.l1 = nn.Linear(784, 2048, bias=True)
        self.mid = nn.Sequential(nn.Linear(2048, 1024, bias=True), nn.BatchNorm1d(1024), nn.ReLU(),
                                 nn.Linear(1024, 512, bias=True), nn.BatchNorm1d(512), nn.ReLU(),
                                 nn.Linear(512, 256, bias=True), nn.BatchNorm1d(256), nn.ReLU()
                                 )
        self.l5 = nn.Linear(256, 10, bias=True)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.mid(x)
        output = F.log_softmax(self.l5(x), dim=1)
        return output


class MNISTTrainer(NNTrainer):
    def _init_nn_model(self):
        self.nn['model'] = MNISTNet()

    def iteration(self, batch):
        inputs = torch.flatten(batch[0].to(self.device['gpu']).float(), 1)
        labels = batch[1].to(self.device['gpu']).long()

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


train_dataset = datasets.MNIST('data', train=True, download=True,
                               transform=transform)
val_dataset = datasets.MNIST('data', train=False,
                             transform=transform)
iter = 64 * 800
train_dataset.data = train_dataset.data[:iter].clone()
train_dataset.target = train_dataset.targets[:iter].clone()

val_dataset.data = val_dataset.data[:iter].clone()
val_dataset.target = val_dataset.targets[:iter].clone()

dataloader_args = {'train': {'dataset': train_dataset},
                   'test': {'dataset': val_dataset}}

if __name__ == "__main__":
    runner = DADTorch(dataloader_args=dataloader_args, args=ap, seed=3, seed_all=True, force=True)
    runner.run(MNISTTrainer)
