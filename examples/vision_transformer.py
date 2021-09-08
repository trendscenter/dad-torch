import glob
import os

import torch
import torch.nn.functional as F
from PIL import Image
from linformer import Linformer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from vit_pytorch.efficient import ViT

from dad_torch import NNTrainer, DADTorch
from dad_torch.metrics import Prf1a

"""Download data from https://www.kaggle.com/c/dogs-vs-cats/data"""

os.makedirs('data', exist_ok=True)
train_dir = 'data/train'
test_dir = 'data/test'
# with zipfile.ZipFile('data/train.zip') as train_zip:
#     train_zip.extractall('data')
#
# with zipfile.ZipFile('data/test.zip') as test_zip:
#     test_zip.extractall('data')

train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))



labels = [path.split('/')[-1].split('.')[0] for path in train_list]

train_list, valid_list = train_test_split(train_list,
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=3)

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

efficient_transformer = Linformer(
    dim=128,
    seq_len=49 + 1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label


class Cifar10Trainer(NNTrainer):
    def _init_nn_model(self):
        self.nn['model'] = ViT(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=2,
            transformer=efficient_transformer,
            channels=3,
        )

    def iteration(self, batch):
        data, label = batch
        inputs = data.to(self.device['gpu'])
        label = label.to(self.device['gpu']).long()

        out = self.nn['model'](inputs)
        loss = F.cross_entropy(out, label)

        _, pred = torch.max(out, 1)
        sc = self.new_metrics()
        sc.add(pred, label.float())

        avg = self.new_averages()
        avg.add(loss.item(), len(inputs))

        return {'loss': loss, 'averages': avg, 'metrics': sc, 'predictions': pred}

    def init_experiment_cache(self):
        self.cache['log_header'] = 'Loss|Accuracy,F1,Precision,Recall'
        self.cache.update(monitor_metric='f1', metric_direction='maximize')

    def new_metrics(self):
        return Prf1a()


lim = 2000
train_data = CatsDogsDataset(train_list[:lim], transform=train_transforms)
valid_data = CatsDogsDataset(valid_list[:int(0.5 * lim)], transform=test_transforms)
test_data = CatsDogsDataset(test_list, transform=test_transforms)
dataloader_args = {
    'train': {'dataset': train_data},
    'validation': {'dataset': valid_data},
    'test': {'dataset': test_data}
}

if __name__ == "__main__":
    print(f"Train Data: {len(train_data)}")
    print(f"Validation Data: {len(valid_data)}")
    print(f"Test Data: {len(test_data)}")

    runner = DADTorch(phase='train',
                      dataloader_args=dataloader_args,
                      seed=3, seed_all=True, force=True, batch_size=32)
    runner.run(Cifar10Trainer)
