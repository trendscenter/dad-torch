import torch as _torch

_SKIP_NORM_Layers = [_torch.nn.BatchNorm1d, _torch.nn.LayerNorm, _torch.nn.GroupNorm]


def _dad_trainable_module(module):
    a_norm_layer = any([isinstance(module, k) for k in _SKIP_NORM_Layers])
    if a_norm_layer:
        return False

    """Has trainable parameters"""
    return len(list(module.parameters())) > 0


class DadHook(_torch.nn.Module):
    def __init__(self, module, device=None, **kw):
        super(DadHook, self).__init__()
        self.module = module
        self.device = device
        self._is_dad_module = {}
        self._reset()

    def _reset(self):
        self.fw_hooks_handle = []
        self.bk_hooks_handle = []
        self._activations = {}
        self._local_grads = {}

    def _hook_fn(self, hook_type, hook_key):
        def get(m, in_grad, out_grad):
            if hook_type.lower() == 'forward':
                for i, b in enumerate(in_grad):
                    if b is not None:
                        self._activations[hook_key] = b
                    break
            if hook_type.lower() == 'backward':
                for i, c in enumerate(out_grad):
                    if c is not None:
                        self._local_grads[hook_key] = c
                    break

        return get

    def _hook(self):
        def _hook_recursive(module_name, module):
            children = list(module.named_children())
            self._is_dad_module[module_name] = False
            if len(children) > 0:
                for children_name, child in children:
                    _hook_recursive(
                        self._hierarchy_key(module_name, children_name),
                        child
                    )

            elif _dad_trainable_module(module):
                self._is_dad_module[module_name] = True
                self.fw_hooks_handle.append(
                    module.register_forward_hook(self._hook_fn('forward', module_name))
                )
                self.bk_hooks_handle.append(
                    module.register_backward_hook(self._hook_fn('backward', module_name))
                )

        if self.training:
            for ch_name, ch in list(self.module.named_children()):
                _hook_recursive(ch_name, ch)

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

    def _hierarchy_key(self, *args):
        return ".".join([f"{a}" for a in args])
