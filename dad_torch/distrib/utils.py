import torch as _torch
<<<<<<< HEAD
from torch import distributed as _dist
from dad_torch.power_iteration_BC import power_iteration_BC
import ujson as json
_GATHER_BROADCAST_BACKENDS = ['mpi', 'gloo']
_ALL_GATHER_BACKENDS = ['nccl']
import os

class DADParallel(_torch.nn.Module):
    def __init__(self, module, device=None, reduction_method='dad', reduction_rank=5, num_pow_iters=1, **kw):
        assert _dist.is_initialized(), "*** Default process group is not initialized. ***"

        super(DADParallel, self).__init__()
=======


class DadHook(_torch.nn.Module):
    def __init__(self, module, device=None, **kw):
        super(DadHook, self).__init__()
>>>>>>> master
        self.module = module
        self.device = device
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
            if len(children) > 0:
                for children_name, child in children:
                    _hook_recursive(
                        self._hierarchy_key(module_name, children_name),
                        child
                    )
            elif len(list(module.parameters())) > 0:
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

<<<<<<< HEAD
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

        try:
            _dist.all_gather(act_gathered, act_tensor)
        except Exception:
            raise(Exception("After _dist.all_gather (act) we have " + str(act_tensor.shape)))
        try:
            _dist.all_gather(grad_gathered, grad_tensor)
        except Exception:
            raise(Exception("After _dist.all_gather (grad) we have " + str(grad_tensor.shape)))

        act_gathered = _torch.cat(act_gathered)
        grad_gathered = _torch.cat(grad_gathered)

        return act_gathered, grad_gathered

    def _dad_reduce_all_at_once(self, queued_communication, *args, **kw):
        """This function plays the role of remote"""
        act_gathered = []
        grad_gathered = []
        for layer in queued_communication.keys():
            act_tensor = queued_communication[layer]["activation"]
            grad_tensor = queued_communication[layer]["delta"]
            act_gathered.extend([_torch.zeros_like(act_tensor) for _ in range(_dist.get_world_size())])
            grad_gathered.extend([_torch.zeros_like(grad_tensor) for _ in range(_dist.get_world_size())])

        _dist.all_gather(act_gathered, act_tensor)
        _dist.all_gather(grad_gathered, grad_tensor)

        act_gathered = _torch.cat(act_gathered)
        grad_gathered = _torch.cat(grad_gathered)

        return act_gathered, grad_gathered

    def dad_backward(self, reduce_in_rank=0, itr=0, all_at_once=False):
        if self.reduction_method == 'base':
            self._base_backward()
        elif self.reduction_method == "dsgd_ar":
            self._dsgd_ar_backward()
        elif self.reduction_method == 'dad':
            self._dad_backward(reduce_in_rank)
        elif self.reduction_method == 'rankdad':
            self._rankdad_backward(reduce_in_rank, itr=itr, all_at_once=all_at_once)
        else:
            raise NotImplementedError(
                f'Not implemented for {self.reduction_method}. Please use one of None, base, dad, OR rankdad.')

    def _base_backward(self, *args, **kwargs):
        size = _dist.get_world_size()
        for param in self.module.parameters():
            grad_gathered = [_torch.zeros_like(param.grad.data) for _ in range(size)]
            _dist.all_gather(grad_gathered, param.grad.data)
            param.grad.data = _torch.stack(grad_gathered).sum(0) / float(size)

    def _dsgd_ar_backward(self, *args, **kwargs):
        size = float(_dist.get_world_size())
        for param in self.module.parameters():
            _dist.all_reduce(param.grad.data, op=_dist.ReduceOp.SUM)
            param.grad.data /= size

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
            try:
                dad_params[f"{layer}.weight"].grad.data = (act_tall.T.mm(local_grad_tall)).T.contiguous()
                if dad_params.get(f"{layer}.bias") is not None:
                    dad_params[f"{layer}.bias"].grad.data = local_grad_tall.sum(0)
            except KeyError:
                continue

    def _rankdad_backward(self, reduce_in_rank=0, itr=0, all_at_once=True):
        all_at_once = True
        dad_params = dict([(k, v) for k, v in self.module.named_parameters()])
        dad_children = dict([(k, v) for k, v in self.module.named_children()])
        queued_communication = {}
        for layer in list(dad_children.keys())[::-1]:
            if str(layer) in ["pos_embedding", "token_embedding"]:
                continue
            act_tall, local_grad_tall = [_torch.ones(1)] * 2
            
            delta_local_reduced, act_local_reduced, benchmarks = power_iteration_BC(
                self._local_grads[layer].T,
                self._activations[layer].T,
                rank=self.reduction_rank,
                numiterations=self.num_pow_iters,
                device=self._local_grads[layer].device,
                do_benchmarks=True
            )
            print("******")
            print(str(layer))
            print("Shapes: " + " _local_grads " + str(self._local_grads[layer].T.shape))
            print("Shapes: " + " _activations " + str(self._activations[layer].T.shape))
            print("Shapes: " + " delta_local_reduced " + str(delta_local_reduced.shape))
            print("Shapes: " + " act_local_reduced " + str(act_local_reduced.shape))
            if all_at_once:
                queued_communication[layer] = dict()
                queued_communication[layer]["delta"] = delta_local_reduced
                queued_communication[layer]["activation"] = act_local_reduced
            else:
                if self.commn_mode == 'all_gather':
                    act_tall, local_grad_tall = self._dad_reduce_all_gather(
                        act_local_reduced.T,
                        delta_local_reduced.T,
                        dest=reduce_in_rank
                    )

                elif self.commn_mode == 'gather_broadcast':
                    act_tall, local_grad_tall = self._dad_reduce_gather_broadcast(
                        act_local_reduced.T,
                        delta_local_reduced.T,
                        dest=reduce_in_rank
                    )
                try:
                    print("Local gradient Shape Before Reassignment: " + str(dad_params[f"{layer}.weight"].grad.data.shape))                    
                    computed = (act_tall.T.mm(local_grad_tall)).T.contiguous()
                    if dad_params[f"{layer}.weight"].grad.data.shape == computed.shape:                        
                        dad_params[f"{layer}.weight"].grad.data = (act_tall.T.mm(local_grad_tall)).T.contiguous()
                        print("Distributed gradient shape After Reassignment: " + str(dad_params[f"{layer}.weight"].grad.data.shape))
                        if dad_params.get(f"{layer}.bias") is not None:
                            dad_params[f"{layer}.bias"].grad.data = local_grad_tall.sum(0)
                except KeyError as e:
                    print("Keyerror in layer " + str(e))
                    continue
        if all_at_once:
            for layer in list(dad_children.keys())[::-1]:
                if str(layer) in ["pos_embedding", "token_embedding"]:
                    continue
                delta_local_reduced = queued_communication[layer]["delta"]
                act_local_reduced = queued_communication[layer]["activation"]
                if self.commn_mode == 'all_gather':
                    act_tall, local_grad_tall = self._dad_reduce_all_gather(
                        act_local_reduced.T,
                        delta_local_reduced.T,
                        dest=reduce_in_rank
                    )

                elif self.commn_mode == 'gather_broadcast':
                    act_tall, local_grad_tall = self._dad_reduce_gather_broadcast(
                        act_local_reduced.T,
                        delta_local_reduced.T,
                        dest=reduce_in_rank
                    )
                try:
                    print("Local gradient Shape Before Reassignment: In layer :" + layer + str(dad_params[f"{layer}.weight"].grad.data.shape))                    
                    computed = (act_tall.T.mm(local_grad_tall)).T.contiguous()
                    if dad_params[f"{layer}.weight"].grad.data.shape == computed.shape:                        
                        dad_params[f"{layer}.weight"].grad.data = (act_tall.T.mm(local_grad_tall)).T.contiguous()
                        print("Distributed gradient shape After Reassignment: " + str(dad_params[f"{layer}.weight"].grad.data.shape))
                        if dad_params.get(f"{layer}.bias") is not None:
                            dad_params[f"{layer}.bias"].grad.data = local_grad_tall.sum(0)

                except KeyError as e:
                    print("Keyerror in layer" + str(e))
                    continue
=======
    def _hierarchy_key(self, *args):
        return ".".join([f"{a}" for a in args])
>>>>>>> master
