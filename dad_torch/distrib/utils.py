import torch as _torch
from torch import distributed as _dist
from dad_torch.power_iteration_BC import power_iteration_BC


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

    """gather is not implemented in nccl backend"""
    # def _dad_reduce(self, act_tensor, grad_tensor, dest=0, *args, **kw):
    #     """This function plays the role of remote"""
    #     act_gathered = [torch.zeros_like(act_tensor) for _ in range(_dist.get_world_size())]
    #     grad_gathered = [torch.zeros_like(grad_tensor) for _ in range(_dist.get_world_size())]
    #
    #     """Compression here"""
    #     _dist.gather(act_tensor, act_gathered if _dist.get_rank() == dest else None, dst=dest)
    #     _dist.gather(grad_tensor, grad_gathered if _dist.get_rank() == dest else None, dst=dest)
    #
    #     if _dist.get_rank() == dest:
    #         act_gathered = _torch.cat(act_gathered)
    #         grad_gathered = _torch.cat(grad_gathered)
    #
    #     _dist.broadcast(act_gathered, src=dest)
    #     _dist.broadcast(grad_gathered, src=dest)
    #     """Decompression here"""
    #
    #     return act_gathered, grad_gathered

    def _dad_reduce(self, act_tensor, grad_tensor, *args, **kw):

        """This function plays the role of remote"""
        act_gathered = [_torch.zeros_like(act_tensor) for _ in range(_dist.get_world_size())]
        grad_gathered = [_torch.zeros_like(grad_tensor) for _ in range(_dist.get_world_size())]

        _dist.all_gather(act_gathered, act_tensor)
        _dist.all_gather(grad_gathered, grad_tensor)

        act_gathered = _torch.cat(act_gathered)
        grad_gathered = _torch.cat(grad_gathered)

        return act_gathered, grad_gathered

    def _rankdad_reduce(self, act_tensor, grad_tensor, *args, **kw):
        """This function plays the role of remote"""
        act_gathered = [_torch.zeros_like(act_tensor) for _ in range(_dist.get_world_size())]
        grad_gathered = [_torch.zeros_like(grad_tensor) for _ in range(_dist.get_world_size())]

        _dist.all_gather(act_gathered, act_tensor)
        _dist.all_gather(grad_gathered, grad_tensor)

        act_gathered = _torch.cat(act_gathered)
        grad_gathered = _torch.cat(grad_gathered)

        return act_gathered, grad_gathered

    def dad_backward(self, reduce_in_rank=0):
        dad_params = dict([(k, v) for k, v in self.module.named_parameters()])
        dad_children = dict([(k, v) for k, v in self.module.named_children()])

        for layer in list(dad_children.keys())[::-1]:
            print(("SHAPE", str(self._local_grads[layer].shape), str(self._activations[layer].shape)), self._local_grads[layer].device, self._activations[layer].device)
            act_tall, local_grad_tall = self._dad_reduce(self._activations[layer], self._local_grads[layer],
                                                         dest=reduce_in_rank)
            print("PRE-REDUCE", act_tall.shape, local_grad_tall.shape)                                                         
            dad_params[f"{layer}.weight"].grad.data = (act_tall.T.mm(local_grad_tall)).T

            if dad_params.get(f"{layer}.bias") is not None:
                dad_params[f"{layer}.bias"].grad.data = local_grad_tall.sum(0)

    def rankdad_backward(self, reduce_in_rank=0):
        reduction_rank=4
        numiterations=1
        #_pp.pprint("IN RANKDAD BACKWARD :)")
        dad_params = dict([(k, v) for k, v in self.module.named_parameters()])
        dad_children = dict([(k, v) for k, v in self.module.named_children()])

        for layer in list(dad_children.keys())[::-1]:
            #print(("SHAPE", str(self._local_grads[layer].shape), str(self._activations[layer].shape)), self._local_grads[layer].device, self._activations[layer].device)
            delta_local_reduced, act_local_reduced = power_iteration_BC(self._local_grads[layer].t(), self._activations[layer].t(), rank=reduction_rank, numiterations=numiterations, device=self._local_grads[layer].device)
            act_local_reduced = act_local_reduced.t()
            delta_local_reduced = delta_local_reduced.t()
            #print("SHAPE-POST", act_local_reduced.shape, delta_local_reduced.shape, act_local_reduced.device, delta_local_reduced.device)
            act_tall, local_grad_tall = self._dad_reduce(act_local_reduced, delta_local_reduced,
                                                         dest=reduce_in_rank)
            #print("PRE-REDUCE", act_tall.shape, local_grad_tall.shape)
            #print(act_tall.shape)
            local_grad_tall, act_tall = power_iteration_BC(local_grad_tall.t(), act_tall.t(), rank=reduction_rank, numiterations=numiterations, device=act_tall.device)   
            act_tall = act_tall.t()
            local_grad_tall = local_grad_tall.t()                                                      
            #print("POST-REDUCE", act_tall.shape, local_grad_tall.shape)
            dad_params[f"{layer}.weight"].grad.data = (act_tall.T.mm(local_grad_tall)).T

            if dad_params.get(f"{layer}.bias") is not None:
                dad_params[f"{layer}.bias"].grad.data = local_grad_tall.sum(0)        

    
