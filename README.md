## DAD
### DAD specific files.
* dad_torch/distrib/utils.py
* dad_torch/trainer.py
* dad-torch/dad-torch.py

#### How to use? Wrap modules as below.

```python
import dad_torch.distrib.utils as utils

model = utils.DADParallel(...) # See trainer.py for the exact usage.
```
#### Run example/MNIST_dadtorch.py as:
1. git clone this repo.
2. RUN cd dad-torch
3. RUN chmod u+x deploy.sh
4. RUN ./deploy.sh
5. RUN cd examples
6. RUN python MNIST_dadtorch.py -ph train -ddp True --dad-reduction True -seed 1 -seed-all True

#### Development ####
Any changes made in the code can be deployed as ./deploy.sh

### Known issues
* gather is not exposed in pytorch DDP backend. So we have used all_gather at the moment.
* UserWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.
  warnings.warn("Using a non-full backward hook when the forward contains multiple autograd Nodes )"
  * full_backward_hook however requires both ends of any layer to be grad enabled, thus it throws an error in layer1 where input is not grad enabled.