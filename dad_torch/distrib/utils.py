import numpy as _np
import torch
import torch as _torch
from torch import distributed as _dist
import os as _os


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

    def __init__(self, module_key, trainer, rank=None):
        super(DADParallel, self).__init__()
        self.module = trainer.nn[module_key]
        self.trainer = trainer

        self.rank = rank
        if not rank:
            self.rank = _dist.get_rank()

        _rank_path = f"{self.rank}" + _os.sep + self.DATA_PATH + _os.sep + module_key
        self.data_path = self.trainer.cache['log_dir'] + _os.sep + _rank_path

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

    def dad_backward(self):
        fk = list(self.trainer.nn.keys())[0]
        dad_layers = [k for k, _ in self.trainer.nn[fk].named_children()][::-1]
        dad_params = dict([(k, v) for k, v in self.trainer.nn[fk].named_parameters()])
        for layer in zip(dad_layers):
            act_file = f"Type-forward_Layer-{layer}_IO-in_Index-0.npy"
            act_tall = self._all_gather_concat(act_file)

            local_grad_file = f"Type-backward_Layer-{layer}_IO-out_Index-0.npy"
            local_grad_tall = self._all_gather_concat(local_grad_file)

            dad_params[layer].grad.data = act_tall.T.mm(local_grad_tall)
