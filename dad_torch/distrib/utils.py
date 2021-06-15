import torch
import torch as _torch
from torch import distributed as _dist


class DADParallel(_torch.nn.Module):

    def __init__(self, module_key, trainer, rank=_dist.get_rank()):
        super(DADParallel, self).__init__()
        self.module = trainer.nn[module_key]
        self.trainer = trainer
        self.device = trainer.device['gpu']
        self.rank = rank
        self._reset()

    def _reset(self):
        self.fw_hooks_handle = []
        self.bk_hooks_handle = []
        self._activations = {}
        self._local_grads = {}

    def _hook_fn(self, rank, hook_type, layer, debug=False):
        if debug:
            print(f"**** {rank}, {hook_type}, {layer} ****")

        def get(m, in_grad, out_grad):
            if hook_type.lower() == 'forward':
                for i, b in enumerate(in_grad):
                    if b is not None:
                        self._activations[layer] = b
                    break
            if hook_type.lower() == 'backward':
                for i, c in enumerate(out_grad):
                    if c is not None:
                        self._local_grads[layer] = c
                    break

        return get

    def _hook(self):
        if self.training:
            for layer, ch in list(self.module.named_children()):
                self.fw_hooks_handle.append(
                    ch.register_forward_hook(self._hook_fn(self.rank, 'forward', layer))
                )
                self.bk_hooks_handle.append(
                    ch.register_backward_hook(self._hook_fn(self.rank, 'backward', layer))
                )

    def _unhook(self):
        for hk in self.fw_hooks_handle:
            hk.remove()
        for hk in self.bk_hooks_handle:
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
        if self.training:
            self._reset()
        output = self.module(*inputs, **kwargs)
        return output

    @staticmethod
    def _all_gather_concat(tensor):
        """This function plays the role of remote"""
        t_list = [_torch.zeros_like(tensor) for _ in _dist.get_world_size()]
        _dist.all_gather(t_list, tensor)
        return _torch.cat(t_list)

    def dad_backward(self):
        dad_params = dict([(k, v) for k, v in self.module.named_parameters()])
        local_grads_tall = torch.Tensor([0])
        for layer, _ in dad_params.keys():
            if 'weight' in layer:
                child_name = layer.split['.'][-2]
                act_tall = self._all_gather_concat(self._activations[child_name])
                local_grad_tall = self._all_gather_concat(self._local_grads[child_name])
                dad_params[layer].grad.data = act_tall.T.mm(local_grad_tall)

            elif 'bias' in layer:
                dad_params[layer].grad.data = local_grads_tall.sum(0, keepdim=True)
                local_grads_tall = torch.Tensor([0])
