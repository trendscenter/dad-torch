import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

from easytorch import EasyTorch, ETTrainer, ConfusionMatrix

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# **Define neural network. I just burrowed from here: https://github.com/pytorch/examples/blob/master/mnist/main.py**
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.l1 = nn.Linear(784, 512, bias=True)
        self.l2 = nn.Linear(512, 256, bias=True)
        self.l3 = nn.Linear(256, 10, bias=True)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        output = F.log_softmax(self.l3(x), dim=1)
        return output


class MNISTTrainer(ETTrainer):
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


if __name__ == "__main__":
    train_dataset = datasets.MNIST('data', train=True, download=True,
                                   transform=transform)
    val_dataset = datasets.MNIST('data', train=False,
                                 transform=transform)

    dataloader_args = {'train': {'dataset': train_dataset},
                       'validation': {'dataset': val_dataset}}
    runner = EasyTorch(dataloader_args=dataloader_args, batch_size=128)
    runner.run(MNISTTrainer)
