import math as _math

import numpy as _np
import torch
import torch as _torch
from torch import distributed as _dist
from torch.utils import data as _data
import os as _os
import glob as _glob


def hook_wrapper(rank, hook_type, layer, save_to='', debug=False):
    if debug:
        print(f"**** {rank}, {hook_type}, {layer} ****")

    name = _os.path.join(save_to, f"Type-{hook_type}_Layer-{layer}")

    def save(m, in_grad, out_grad):
        if hook_type.lower() == 'forward':
            for i, b in enumerate(in_grad):
                if b is not None:
                    _np.save(name + f"_IO-in_Index-{i}.npy", b.clone().detach().numpy())
                break
        if hook_type.lower() == 'backward':
            for i, c in enumerate(out_grad):
                if c is not None:
                    _np.save(name + f"_IO-out_Index-{i}.npy", c.clone().detach().numpy())
                break

    return save


class DADParallel(_torch.nn.Module):
    DATA_PATH = '_dad_layers_data_'
    GRADS_PATH = '_dad_weights_update_'

    def __init__(self, module_key, trainer, rank=None):
        super(DADParallel, self).__init__()
        self.module = trainer.nn[module_key]
        self.trainer = trainer

        self.rank = rank
        if not rank:
            self.rank = _dist.get_rank()

        self.data_path = self.trainer.cache[
                             'log_dir'] + _os.sep + f"{self.rank}" + _os.sep + self.DATA_PATH + _os.sep + module_key
        self.grads_path = self.trainer.cache[
                              'log_dir'] + _os.sep + f"{self.rank}" + _os.sep + self.GRADS_PATH + _os.sep + module_key
        self.fw_hooks = []
        self.bk_hooks = []

    def _hook(self):
        if self.training:
            for model_key in self.trainer.nn.keys():
                for layer, ch in list(self.trainer.nn[model_key].named_children()):
                    self.fw_hooks.append(
                        ch.register_forward_hook(hook_wrapper(self.state['clientId'], 'forward', layer, self.data_path))
                    )
                    self.bk_hooks.append(
                        ch.register_backward_hook(
                            hook_wrapper(self.state['clientId'], 'backward', layer, self.data_path))
                    )

    def _unhook(self):
        for hk in self.fw_hooks:
            hk.remove()
        for hk in self.bk_hooks:
            hk.remove()

    def train(self, mode=True):
        self.module.train(mode)
        self._hook()
        return self

    def eval(self):
        self.module.eval()
        self._unhook()
        return self

    def forward(self, *inputs, **kwargs):
        output = self.module(*inputs, **kwargs)
        return output

    def _all_gather_concat(self, file):
        """This function plays the role of remote"""
        t = torch.Tensor(_np.load(file)).to(self.trainer.device['gpu'])
        t_list = [_torch.zeros_like(t) for _ in _dist.get_world_size()]
        _dist.all_gather(t_list, t)
        return _torch.cat(t_list)

    def backward(self):
        fk = list(self.trainer.nn.keys())[0]
        dad_layers = [k for k, _ in self.trainer.nn[fk].named_children()][::-1]
        dad_params = dict([(k, v) for k, v in self.trainer.nn[fk].named_parameters()])
        for layer in zip(dad_layers):
            act_file = f"Type-forward_Layer-{layer}_IO-in_Index-0.npy"
            act_tall = self._all_gather_concat(act_file)

            local_grad_file = f"Type-backward_Layer-{layer}_IO-out_Index-0.npy"
            local_grad_tall = self._all_gather_concat(local_grad_file)

            dad_params[layer].grad.data = act_tall.T.mm(local_grad_tall)


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
