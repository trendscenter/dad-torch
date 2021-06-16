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
6. RUN python MNIST_dadtorch.py -ph train -dad True -seed 1

#### Development ####
Any changes made in the code can be deployed as ./deploy.sh
