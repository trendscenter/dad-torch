import torch as _torch
import torch.nn.functional as _F
from torch import distributed as _dist

from dad_torch.utils import power_iteration_BC as _power_iter_BC
from .utils import DadHook as _DADHook


class DADParallel(_DADHook):
    def __init__(self, module, device=None, reduction_method='dad', reduction_rank=5, num_pow_iters=1, **kw):
        assert _dist.is_initialized(), "*** Default process group is not initialized. ***"

        super(DADParallel, self).__init__(module=module, device=device, **kw)
        self.cache = kw.get('cache', {})
        self.reduction_method = reduction_method
        self.reduction_rank = reduction_rank
        self.num_pow_iters = num_pow_iters
        self.commn_mode = kw.get('commn_mode', 'all_gather')

    """gather is not implemented in nccl backend"""

    def _dad_reduce_gather_broadcast(self, act_tensor, grad_tensor, dest=0, *args, **kw):
        """This function plays the role of remote"""
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

    def _dad_reduce_all_gather(self, act_tensor, grad_tensor, *args, **kw):
        """This function plays the role of remote"""
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

    def _synced_param_update_(self, activations, local_grads, parameters, dst_rank, *args):
        act_tall, local_grad_tall = [_torch.ones(1)] * 2
        if self.commn_mode == 'all_gather':
            act_tall, local_grad_tall = self._dad_reduce_all_gather(
                activations,
                local_grads,
                dest=dst_rank
            )

        elif self.commn_mode == 'gather_broadcast':
            act_tall, local_grad_tall = self._dad_reduce_gather_broadcast(
                activations,
                local_grads,
                dest=dst_rank
            )

        parameters["weight"].grad.data = (act_tall.T.mm(local_grad_tall)).T.contiguous()
        if parameters.get("bias") is not None:
            parameters["bias"].grad.data = local_grad_tall.sum(0)

    def _dad_backward(self, reduce_in_rank=0):

        def _backward(module_name, module):
            dad_params = dict(list(module.named_parameters())[::-1])
            dad_children = dict(list(module.named_children())[::-1])

            if len(dad_children) > 0:
                for child_name, child in dad_children.items():
                    _backward(self._hierarchy_key(module_name, child_name), child)

            elif self._is_dad_module.get(module_name):
                """ Update and sync weights """
                self._synced_param_update_(
                    self._activations[module_name],
                    self._local_grads[module_name],
                    dad_params,
                    reduce_in_rank,
                    module_name,
                )

        for ch_name, ch in list(self.module.named_children())[::-1]:
            _backward(ch_name, ch)

    def _rankdad_backward(self, reduce_in_rank=0):

        def _backward(module_name, module):
            dad_params = dict(list(module.named_parameters())[::-1])
            dad_children = dict(list(module.named_children())[::-1])

            if len(dad_children) > 0:
                for child_name, child in dad_children.items():
                    _backward(self._hierarchy_key(module_name, child_name), child)

            elif self._is_dad_module.get(module_name):
                """ Rank reduce with PowerIteration """
                delta_local_reduced, act_local_reduced = _power_iter_BC(
                    self._local_grads[module_name].T,
                    self._activations[module_name].T,
                    rank=self.reduction_rank,
                    numiterations=self.num_pow_iters,
                    device=self._local_grads[module_name].device
                )
                """Rank-reduction end. """

                """ Pick Max rank of the world and pad to match """
                max_rank = _torch.Tensor([delta_local_reduced.shape[1]]).to(self.device)
                _dist.all_reduce(max_rank, _dist.ReduceOp.MAX)

                if max_rank > delta_local_reduced.shape[1]:
                    _pad = (0, int(max_rank.item() - delta_local_reduced.shape[1]))
                    act_local_reduced = _F.pad(act_local_reduced, _pad)
                    delta_local_reduced = _F.pad(delta_local_reduced, _pad)
                """ Padding End """

                """ Update and sync weights """
                self._synced_param_update_(
                    act_local_reduced.T,
                    delta_local_reduced.T,
                    dad_params,
                    reduce_in_rank
                )

        for ch_name, ch in list(self.module.named_children())[::-1]:
            _backward(ch_name, ch)
