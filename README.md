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
6. RUN python MNIST_dadtorch.py -ph train -ddp True --dad-reduction True -seed 1 -seed-all True -log net_logs_DADs2b64

#### Development ####
Any changes made in the code can be deployed as ./deploy.sh

### Plotting graphs
Use runtime_plotter.py in examples to compare different runtimes as below.

* cd examples
* python runtime_plotter.py -paths <List of paths to logs folder> -keys <list of keys to plot>
  * Example: python runtime_plotter.py -paths nest_logs_DAD_BCs4b128  net_logs_DDPs4b128 -keys batch_duration
   Passing nothing to -keys will plot everything duration of forward, backward, each layers ... (not recommended)
  * List of duration keys used are in training_iteration(self, i, batch) -> dict: method of training.py.
  * Example experiment schedule is in examples/jobs.py
* Hint: To generate plot for experiment with 4 sites, and batch size 128 one can run
  * python runtime_plotter.py -paths `ls -d *s4*b128*` -keys batch_duration
* All experiments data are in /home/akhanal1/TrendsLab/dad-torch/examples for referenceThere will be error if 

**Plotter will throw an exception when a key is not available in all provided paths. For example, we cannot plot key dad_backward_duration for list of experiments with DDP(standard SGD) because it will not have that key, only DAD experiment will have it. So it only makes sense to only plot batch_duration for such case.**
   
Notes: When running experiments(as in point 6 above) we should name the -log folder in such a way that the last element after splitting the directory name with _(underscore) represents the 
experiment and will be used to label the color legend in the plot. Example: net_logs_DDPs4b128 will have DDPs4b128 as the legend in the plot, similarly for net_logs_DAD_BC-IBs4b128, the color legend label will be BC-IBs4b128.


### Known issues
* gather is not exposed in pytorch DDP backend. So we have used all_gather at the moment.
* UserWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.
  warnings.warn("Using a non-full backward hook when the forward contains multiple autograd Nodes )"
  * full_backward_hook however requires both ends of any layer to be grad enabled, thus it throws an error in layer1 where input is not grad enabled.