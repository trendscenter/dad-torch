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
