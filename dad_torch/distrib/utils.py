import torch as _torch
from torch import distributed as _dist


class DADParallel(_torch.nn.Module):
    def __init__(self, module, device=None, rank=None, **kw):
        assert _dist.is_initialized(), "*** Default process group is not initialized. ***"

        self.rank = rank
        if self.rank is None:
            self.rank = _dist.get_rank()

        self.debug = self.rank == 0
        super(DADParallel, self).__init__()
        self.module = module
        self.device = device
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

    def _all_gather_concat(self, tensor, *args, **kw):
        """This function plays the role of remote"""
        t_list = [_torch.zeros_like(tensor) for _ in range(_dist.get_world_size())]
        _dist.all_gather(t_list, tensor)
        t_list = _torch.cat(t_list)
        return t_list

    def dad_backward(self):
        dad_params = dict([(k, v) for k, v in self.module.named_parameters()])
        dad_children = dict([(k, v) for k, v in self.module.named_children()])

        for layer in list(dad_children.keys())[::-1]:
            act_tall = self._all_gather_concat(self._activations[layer], layer=layer, mode='act')
            local_grad_tall = self._all_gather_concat(self._local_grads[layer], layer=layer, mode='grad')
            dad_params[f"{layer}.weight"].grad.data = (act_tall.T.mm(local_grad_tall)).T

            if dad_params.get(f"{layer}.bias") is not None:
                dad_params[f"{layer}.bias"].grad.data = local_grad_tall.sum(0)
