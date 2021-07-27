import torch
import torch as _torch
from torch import distributed as _dist
from dad_torch.power_iteration_BC import power_iteration_BC

_GATHER_BROADCAST_BACKENDS = ['mpi', 'gloo']
_ALL_GATHER_BACKENDS = ['nccl']


class DADParallel(_torch.nn.Module):
    def __init__(self, module, device=None, reduction_method='dad', reduction_rank=5, num_pow_iters=1, **kw):
        assert _dist.is_initialized(), "*** Default process group is not initialized. ***"

        super(DADParallel, self).__init__()
        self.module = module
        self.device = device
        self.cache = kw.get('cache', {})
        self.reduction_method = reduction_method
        self.reduction_rank = reduction_rank
        self.num_pow_iters = num_pow_iters
        self.commn_mode = kw.get('commn_mode', 'all_gather')
        self._reset()

    def _reset(self):
        self.fw_hooks_handle = []
        self.bk_hooks_handle = []
        self._activations = {}
        self._local_grads = {}

    def _hook_fn(self, hook_type, layer):
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
                    ch.register_forward_hook(self._hook_fn('forward', layer))
                )
                self.bk_hooks_handle.append(
                    ch.register_backward_hook(self._hook_fn('backward', layer))
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

    """gather is not implemented in nccl backend"""

    def _dad_reduce_gather_broadcast(self, act_tensor, grad_tensor, dest=0, rank_sizes=None, *args, **kw):
        """This function plays the role of remote"""
        if rank_sizes:
            act_gathered = [
                _torch.zeros((r, act_tensor.shape[1]), dtype=act_tensor.dtype, device=act_tensor.device) for
                r in rank_sizes]
            grad_gathered = [
                _torch.zeros((r, grad_tensor.shape[1]), dtype=grad_tensor.dtype, device=grad_tensor.device)
                for r in rank_sizes]
        else:
            act_gathered = [_torch.zeros_like(act_tensor) for _ in range(_dist.get_world_size())]
            grad_gathered = [_torch.zeros_like(grad_tensor) for _ in range(_dist.get_world_size())]

        """Compression here"""
        _dist.gather(act_tensor, act_gathered if _dist.get_rank() == dest else None, dst=dest)
        _dist.gather(grad_tensor, grad_gathered if _dist.get_rank() == dest else None, dst=dest)

        act_gathered = _torch.cat(act_gathered)
        grad_gathered = _torch.cat(grad_gathered)

        _dist.broadcast(act_gathered, src=dest)
        _dist.broadcast(grad_gathered, src=dest)
        """Decompression here"""

        return act_gathered, grad_gathered

    def _dad_reduce_all_gather(self, act_tensor, grad_tensor, rank_sizes=None, *args, **kw):

        """This function plays the role of remote"""
        if rank_sizes:
            act_gathered = [
                _torch.zeros((r, act_tensor.shape[1]), dtype=act_tensor.dtype, device=act_tensor.device) for
                r in rank_sizes]
            grad_gathered = [
                _torch.zeros((r, grad_tensor.shape[1]), dtype=grad_tensor.dtype, device=grad_tensor.device)
                for r in rank_sizes]
        else:
            act_gathered = [_torch.zeros_like(act_tensor) for _ in range(_dist.get_world_size())]
            grad_gathered = [_torch.zeros_like(grad_tensor) for _ in range(_dist.get_world_size())]

        _dist.all_gather(act_gathered, act_tensor)
        _dist.all_gather(grad_gathered, grad_tensor)

        act_gathered = _torch.cat(act_gathered)
        grad_gathered = _torch.cat(grad_gathered)

        return act_gathered, grad_gathered

    def dad_backward(self, reduce_in_rank=0):
        if self.reduction_method == 'base':
            self._base_backward()
        elif self.reduction_method == 'dad':
            self._dad_backward(reduce_in_rank)
        elif self.reduction_method == 'rankdad':
            self._rankdad_backward(reduce_in_rank)
        else:
            raise NotImplementedError(
                f'Not implemented for {self.reduction_method}. Please use one of None, base, dad, OR rankdad.')

    def _base_backward(self, *args, **kwargs):
        size = _dist.get_world_size()
        for param in self.module.parameters():
            grad_gathered = [_torch.zeros_like(param.grad.data) for _ in range(size)]
            _dist.all_gather(grad_gathered, param.grad.data)
            param.grad.data = _torch.stack(grad_gathered).sum(0) / float(size)

    def _dad_backward(self, reduce_in_rank=0):
        dad_params = dict([(k, v) for k, v in self.module.named_parameters()])
        dad_children = dict([(k, v) for k, v in self.module.named_children()])

        for layer in list(dad_children.keys())[::-1]:
            act_tall, local_grad_tall = [_torch.ones(1)] * 2

            if self.commn_mode == 'all_gather':
                act_tall, local_grad_tall = self._dad_reduce_all_gather(
                    self._activations[layer],
                    self._local_grads[layer],
                    dest=reduce_in_rank
                )
            elif self.commn_mode == 'gather_broadcast':

                act_tall, local_grad_tall = self._dad_reduce_gather_broadcast(
                    self._activations[layer],
                    self._local_grads[layer],
                    dest=reduce_in_rank
                )

            """Update weights"""
            dad_params[f"{layer}.weight"].grad.data = (act_tall.T.mm(local_grad_tall)).T.contiguous()
            if dad_params.get(f"{layer}.bias") is not None:
                dad_params[f"{layer}.bias"].grad.data = local_grad_tall.sum(0)

    def _rankdad_backward(self, reduce_in_rank=0):
        dad_params = dict([(k, v) for k, v in self.module.named_parameters()])
        dad_children = dict([(k, v) for k, v in self.module.named_children()])

        for layer in list(dad_children.keys())[::-1]:
            act_tall, local_grad_tall = [_torch.ones(1)] * 2

            delta_local_reduced, act_local_reduced = power_iteration_BC(
                self._local_grads[layer].T,
                self._activations[layer].T,
                rank=self.reduction_rank,
                numiterations=self.num_pow_iters,
                device=self._local_grads[layer].device
            )

            """Effective ranks of the world."""
            effective_world_ranks = [torch.Tensor([self.reduction_rank]).to(self.device) for _ in
                                     range(_dist.get_world_size())]
            _dist.all_gather(effective_world_ranks, torch.Tensor([delta_local_reduced.shape[1]]).to(self.device))
            effective_world_ranks = [int(r.item()) for r in effective_world_ranks]

            if self.commn_mode == 'all_gather':
                act_tall, local_grad_tall = self._dad_reduce_all_gather(
                    act_local_reduced.T,
                    delta_local_reduced.T,
                    dest=reduce_in_rank,
                    rank_sizes=effective_world_ranks
                )

            elif self.commn_mode == 'gather_broadcast':
                act_tall, local_grad_tall = self._dad_reduce_gather_broadcast(
                    act_local_reduced.T,
                    delta_local_reduced.T,
                    dest=reduce_in_rank,
                    rank_sizes=effective_world_ranks
                )

            dad_params[f"{layer}.weight"].grad.data = (act_tall.T.mm(local_grad_tall)).T.contiguous()
            if dad_params.get(f"{layer}.bias") is not None:
                dad_params[f"{layer}.bias"].grad.data = local_grad_tall.sum(0)
