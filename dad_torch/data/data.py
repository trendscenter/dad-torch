import json as _json
import multiprocessing as _mp
import os as _os
from collections import Callable
from functools import partial as _partial
from os import sep as _sep

import numpy as _np
import torch as _torch
import torch.utils.data as _data
from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
from torch.utils.data._utils.collate import default_collate as _default_collate

import dad_torch.data.datautils as _du
import dad_torch.utils as _etutils
from dad_torch.utils.logger import *
from .datautils import UnPaddedDDPSampler


def _job(total, func, i, f):
    print(f"Working on: [ {i}/{total} ]", end='\r')
    return func(f)


def multiRun(nproc: int, data_list: list, func: Callable) -> list:
    _files = []
    for ix, file in enumerate(data_list, 1):
        _files.append([ix, file])

    with _mp.Pool(processes=nproc) as pool:
        return list(
            pool.starmap(_partial(_job, len(_files), func), _files)
        )


def safe_collate(batch):
    r"""Safely select batches/skip dataset_cls(errors in file loading."""
    return _default_collate([b for b in batch if b])


def num_workers(args, loader_args, distributed=False):
    if distributed:
        return (loader_args['num_workers'] + args['num_gpus'] - 1) // args['num_gpus']
    return loader_args['num_workers']


def batch_size(args, loader_args, distributed=False):
    if distributed:
        loader_args['batch_size'] = loader_args['batch_size'] // args['num_gpus']
    return loader_args['batch_size']


def _seed_worker(worker_id):
    seed = (int(_torch.initial_seed()) + worker_id) % (2 ** 32 - 1)
    _np.random.seed(seed)


def _et_data_job_func(mode, file, dataspec, args, dataset_cls):
    test_dataset = dataset_cls(mode=mode, **args)
    test_dataset.add(files=[file], verbose=False, **dataspec)
    return test_dataset


def _et_data_job(mode, arg, dspec, cls, total, func, verbose, i, file):
    if verbose:
        print(f"Working on: [ {i} / {total} ]", end='\r')
    return func(mode, file, dspec, arg, cls)


class ETDataHandle:

    def __init__(self, args=None, dataloader_args=None, **kw):
        self.dataset = {}
        self.dataloader = {}
        self.args = _etutils.FrozenDict(args)
        self.dataloader_args = _etutils.FrozenDict(dataloader_args)

    def get_dataset(self, handle_key, files, dataspec: dict, reuse=False, dataset_cls=None):
        if reuse and self.dataset.get(handle_key):
            return self.dataset[handle_key]

        dataset = dataset_cls(mode=handle_key, limit=self.args['load_limit'], **self.args)
        dataset.add(files=files, verbose=self.args['verbose'], **dataspec)
        self.dataset[handle_key] = dataset
        return dataset

    def get_train_dataset(self, split_file, dataspec: dict, dataset_cls=None):
        if dataset_cls is None or self.dataloader_args.get('train', {}).get('dataset'):
            return self.dataloader_args.get('train', {}).get('dataset')

        r"""Load the train data from current fold/split."""
        with open(dataspec['split_dir'] + _sep + split_file) as file:
            split = _json.loads(file.read())
            train_dataset = self.get_dataset('train', split.get('train', []),
                                             dataspec, dataset_cls=dataset_cls)
            return train_dataset

    def get_validation_dataset(self, split_file, dataspec: dict, dataset_cls=None):
        if dataset_cls is None or self.dataloader_args.get('validation', {}).get('dataset'):
            return self.dataloader_args.get('validation', {}).get('dataset')

        r""" Load the validation data from current fold/split."""
        with open(dataspec['split_dir'] + _sep + split_file) as file:
            split = _json.loads(file.read())
            val_dataset = self.get_dataset('validation', split.get('validation', []),
                                           dataspec, dataset_cls=dataset_cls)
            if val_dataset and len(val_dataset) > 0:
                return val_dataset

    def get_test_dataset(self, split_file, dataspec: dict, dataset_cls=None):
        if dataset_cls is None or self.dataloader_args.get('test', {}).get('dataset'):
            return self.dataloader_args.get('test', {}).get('dataset')

        with open(dataspec['split_dir'] + _sep + split_file) as file:
            _files = _json.loads(file.read()).get('test', [])[:self.args['load_limit']]
            if self.args['load_sparse'] and len(_files) > 1:
                datasets = ETDataHandle.multi_load('test', _files, dataspec, self.args, dataset_cls)
                success(f'\n{len(datasets)} sparse dataset loaded.', self.args['verbose'])
            else:
                datasets = self.get_dataset('test', _files, dataspec, dataset_cls=dataset_cls)

            if len(datasets) > 0 and sum([len(t) for t in datasets if t]) > 0:
                return datasets

    def get_loader(self,
                   handle_key='', distributed=False,
                   use_unpadded_sampler=False,
                   reuse=False, **kw
                   ):

        if reuse and self.dataloader.get(handle_key) is not None:
            return self.dataloader[handle_key]

        args = {**self.args}
        args['distributed'] = distributed
        args['use_unpadded_sampler'] = use_unpadded_sampler
        args.update(self.dataloader_args.get(handle_key, {}))
        args.update(**kw)

        loader_args = {
            'dataset': None,
            'batch_size': 1,
            'sampler': None,
            'shuffle': False,
            'batch_sampler': None,
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': _seed_worker if args.get('seed_all') else None
        }
        for k in loader_args.keys():
            loader_args[k] = args.get(k, loader_args.get(k))

        if args['distributed']:
            sampler_args = {
                'num_replicas': args.get('replicas'),
                'rank': args.get('rank'),
                'shuffle': args.get('shuffle'),
                'seed': args.get('seed')
            }

            if loader_args.get('sampler') is None:
                loader_args['shuffle'] = False  # Shuffle is mutually exclusive with sampler
                if args['use_unpadded_sampler']:
                    loader_args['sampler'] = UnPaddedDDPSampler(loader_args['dataset'], **sampler_args)
                else:
                    loader_args['sampler'] = _data.distributed.DistributedSampler(loader_args['dataset'],
                                                                                  **sampler_args)

            loader_args['num_workers'] = num_workers(args, loader_args, True)
            loader_args['batch_size'] = batch_size(args, loader_args, True)

        self.dataloader[handle_key] = _DataLoader(collate_fn=safe_collate, **loader_args)
        return self.dataloader[handle_key]

    def create_splits(self, dataspec, out_dir):
        if _du.should_create_splits_(out_dir, dataspec, self.args):
            _du.default_data_splitter_(dspec=dataspec, args=self.args)
            info(f"{len(_os.listdir(dataspec['split_dir']))} split(s) created in '{dataspec['split_dir']}' directory.",
                 self.args['verbose'])
        else:
            splits_len = len(_os.listdir(dataspec['split_dir']))
            info(f"{splits_len} split(s) loaded from '{dataspec['split_dir']}' directory.",
                 self.args['verbose'] and splits_len > 0)

    def init_dataspec_(self, dataspec: dict):
        for k in dataspec:
            if '_dir' in k:
                path = _os.path.join(self.args['dataset_dir'], dataspec[k])
                path = path.replace(f"{_sep}{_sep}", _sep)
                if path.endswith(_sep):
                    path = path[:-1]
                dataspec[k] = path

    @staticmethod
    def multi_load(mode, files, dataspec, args, dataset_cls, func=_et_data_job_func) -> list:

        r"""Note: Only works with dad_torch's default args from dad_torch import args"""
        _files = []
        for ix, f in enumerate(files, 1):
            _files.append([ix, f])

        nw = min(num_workers(args, args, args['use_ddp']), len(_files))
        with _mp.Pool(processes=max(1, nw)) as pool:
            return list(
                pool.starmap(
                    _partial(_et_data_job, mode, args, dataspec, dataset_cls, len(_files), func, args['verbose']),
                    _files)
            )


class ETDataset(_Dataset):
    def __init__(self, mode='init', limit=None, **kw):
        self.mode = mode
        self.limit = limit
        self.indices = []
        self.data = {}

        self.args = _etutils.FrozenDict(kw)
        self.dataspecs = _etutils.FrozenDict({})

    def load_index(self, dataset_name, file):
        r"""
        Logic to load indices of a single file.
        -Sometimes one image can have multiple indices like U-net where we have to get multiple patches of images.
        """
        self.indices.append([dataset_name, file])

    def _load_indices(self, dataspec_name, files, verbose=True):
        r"""
        We load the proper indices/names(whatever is called) of the files in order to prepare minibatches.
        Only load lim numbr of files so that it is easer to debug(Default is infinite, -lim/--load-lim argument).
        """
        _files = files[:self.limit]
        if len(_files) > 1:
            dataset_objs = ETDataHandle.multi_load(
                self.mode, _files, self.dataspecs[dataspec_name], self.args, self.__class__
            )
            self.gather(dataset_objs)
        else:
            for f in _files:
                self.load_index(dataspec_name, f)

        success(f'\n{dataspec_name}, {self.mode}, {len(self)} indices Loaded.', verbose)

    def gather(self, dataset_objs):
        for d in dataset_objs:
            attributes = vars(d)
            for k, v in attributes.items():
                if isinstance(v, _etutils.FrozenDict):
                    continue

                if isinstance(v, list):
                    self.__getattribute__(f"{k}").extend(v)

                elif isinstance(attributes[f"{k}"], dict):
                    self.__getattribute__(f"{k}").update(**v)

                elif isinstance(attributes[f"{k}"], set):
                    self.__getattribute__(f"{k}").union(v)

    def __getitem__(self, index):
        r"""
        Logic to load one file and send to model. The mini-batch generation will be handled by Dataloader.
        Here we just need to write logic to deal with single file.
        """
        raise NotImplementedError('Must be implemented by child class.')

    def __len__(self):
        return len(self.indices)

    def transforms(self, **kw):
        return None

    def add(self, files, verbose=True, **kw):
        r""" An extra layer for added flexibility."""
        self.dataspecs[kw['name']] = kw
        self._load_indices(dataspec_name=kw['name'], files=files, verbose=verbose)
