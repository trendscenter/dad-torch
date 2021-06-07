import glob as _glob
import json as _json
import math as _math
import os as _os
import random as _rd

import numpy as _np
import torch as _torch
from torch import distributed as _dist
from torch.utils import data as _data

_sep = _os.sep


def create_ratio_split(files, save_to_dir=None, ratio: list = None, first_key='train', name='SPLIT',
                       shuffle_files=True):
    if shuffle_files:
        _rd.shuffle(files)

    keys = [first_key]
    if len(ratio) == 2:
        keys.append('test')
    elif len(ratio) == 3:
        keys.append('validation')
        keys.append('test')

    _ratio = ratio[::-1]
    locs = _np.array([sum(_ratio[0:i + 1]) for i in range(len(ratio) - 1)])
    locs = (locs * len(files)).astype(int)
    splits = _np.split(files[::-1], locs)[::-1]
    splits = dict([(k, sp.tolist()[::-1]) for k, sp in zip(keys, splits)])
    if save_to_dir:
        f = open(save_to_dir + _sep + f'{name}.json', "w")
        f.write(_json.dumps(splits))
        f.close()
    else:
        return splits


def create_k_fold_splits(files, k=0, save_to_dir=None, shuffle_files=True, name='SPLIT'):
    if shuffle_files:
        _rd.shuffle(files)

    ix_splits = _np.array_split(_np.arange(len(files)), k)
    for i in range(len(ix_splits)):
        test_ix = ix_splits[i].tolist()
        val_ix = ix_splits[(i + 1) % len(ix_splits)].tolist()
        train_ix = [ix for ix in _np.arange(len(files)) if ix not in test_ix + val_ix]

        splits = {'train': [files[ix] for ix in train_ix],
                  'validation': [files[ix] for ix in val_ix],
                  'test': [files[ix] for ix in test_ix]}

        if save_to_dir:
            f = open(save_to_dir + _sep + f"{name}_{i}.json", "w")
            f.write(_json.dumps(splits))
            f.close()
        else:
            return splits


def uniform_mix_two_lists(smaller, larger, shuffle=True):
    if shuffle:
        _rd.shuffle(smaller)
        _rd.shuffle(larger)

    len_smaller, len_larger = len(smaller), len(larger)

    accumulator = []
    while len(accumulator) < len_smaller + len_larger:
        try:
            for i in range(int(len_larger / len_smaller)):
                accumulator.append(larger.pop())
        except Exception:
            pass
        try:
            accumulator.append(smaller.pop())
        except Exception:
            pass

    return accumulator


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def should_create_splits_(log_dir, dspec, args):
    if dspec.get('split_dir') and _os.path.exists(dspec.get('split_dir')) and len(list(
            _os.listdir(dspec.get('split_dir')))) > 0:
        return False

    dspec['split_dir'] = log_dir + _sep + 'splits'
    if _os.path.exists(dspec['split_dir']) and len(list(_os.listdir(dspec['split_dir']))) > 0:
        return False

    _os.makedirs(dspec['split_dir'], exist_ok=True)
    if args['num_folds'] is None and args['split_ratio'] is None:
        with open(dspec['split_dir'] + _sep + 'experiment.json', 'w') as sp:
            sp.write(_json.dumps({'train': [], 'validation': [], 'test': []}))
        return False

    return True


def list_files(dspec):
    ext = dspec.get('extension', '*').replace('.', '')
    rec = dspec.get('recursive', False)
    rec_pattern = '**/' if rec else ''
    if dspec.get('sub_folders') is None:
        path = dspec['data_dir']
        return [f.replace(path + _sep, '') for f in
                _glob.glob(f"{path}/{rec_pattern}*.{ext}", recursive=rec)]

    files = []
    for sub in dspec['sub_folders']:
        path = dspec['data_dir'] + _sep + sub
        files += [f.replace(dspec['data_dir'] + _sep, '') for f in
                  _glob.glob(f"{path}/{rec_pattern}*.{ext}", recursive=rec)]
    return files


def default_data_splitter_(dspec, args):
    r"""
    Initialize k-folds for given dataspec.
        If: custom splits path is given it will use the splits from there
        else: will create new k-splits and run k-fold cross validation.
    """
    if args.get('num_folds') is not None:
        create_k_fold_splits(
            files=list_files(dspec),
            k=args['num_folds'],
            save_to_dir=dspec['split_dir'],
            name=dspec['name']
        )
    elif args.get('split_ratio') is not None:
        create_ratio_split(
            files=list_files(dspec),
            save_to_dir=dspec['split_dir'],
            ratio=args['split_ratio'],
            name=dspec['name']
        )


class UnPaddedDDPSampler(_data.Sampler):
    r"""fork from official pytorch repo: torch.data.distributed.DistributedSampler where padding is off"""
    r"""https://github.com/pytorch/"""

    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not _dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = _dist.get_world_size()
        if rank is None:
            if not _dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = _dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(_math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = _torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = _torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        """Do not pad anything"""
        # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
