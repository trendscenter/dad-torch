from torch.utils.data.dataset import Dataset
from torchtext import data, datasets, vocab
import torch
import numpy as np


class ImdbDataset(Dataset):
    def __init__(self, train=False, train_dataset=None, test_dataset=None):
        super(ImdbDataset, self).__init__()
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
        LABEL = data.Field(sequential=False)
        if train_dataset is not None:
            self.train = train_dataset
            self.test = test_dataset
        else:
            self.train, self.test = datasets.IMDB.splits(TEXT, LABEL)
        # self.train, self.test = tdata.split(split_ratio=0.8)
        # self.train, self.test = self.train.split(split_ratio=0.9)

        if train:
            self.dataset = self.train
        else:
            self.dataset = self.test
        TEXT.build_vocab(
            self.train, max_size=50_000 - 2
        )  # - 2 to make space for <unk> and <pad>
        LABEL.build_vocab(self.train)
        self.TEXT = TEXT
        self.LABEL = LABEL

    def __getitem__(self, index: int):
        return self.dataset[index].text, int(self.dataset[index].label == "pos")

    def __len__(self):
        return len(self.dataset)

    def get_subset_by_indices(self, indices):
        return ImdbDataset(
            train=True,
            train_dataset=data.Dataset(
                [self.train[i] for i in indices], self.train.fields
            ),
        )
