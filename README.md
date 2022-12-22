
## Code Accompanying "A Risk-Sensitive Approach to Policy Optimization"

----------------------

----------------------

### Building & Installing

 To install this code and get it running, first set up a python virtual environment. We developed and tested this using 
 a conda environment and python 3.6, but one should be able to replicate the results using other configurations.  Create the 
 environment and then do the following:
 
1. Clone this repository.

2.  This code base uses PyTorch.  To ensure that you get the correct version, go to their 
[installation page](https://pytorch.org/get-started/locally/) and enter your system characteristics.

3. If you want to run Safety Gym experiments, clone their repository and place it in the `./envs` folder.
    - Running Safety Gym requires MuJoCo. Further instructions for downloading and installing MuJoco can be found at
      [their website](http://www.mujoco.org).
    - Install Safety Gym by going to `./envs/safety-gym` and typing `pip install -e .`.
 
4. The parallelization in this code base relies on MPI.  We found that the conda-forge version works well;
to get it, type `conda install -c conda-forge mpi4py`.

5. Install risk and its subpackages.  From this home folder, type `pip install -e .`.

----------------------

### Repository Notes

- The configuration files used to generate the results in the paper are in `./configs.`  Instructions for running are below.

- The `./risk/common` folder has pieces useful to multiple packages and algorithms, while the algorithms in the other packages are more specific.

- The PolicyOptimizer class (`./risk/rl/policy_optimizer.py`) is a configurable on-policy learner.  To have it to act as PPO in the style of Spinning Up,
make sure to turn on generalized advantage estimation, the surrogate objective, and KL-based early stopping in the configuration.
  
- Our ConstrainedPolicyOptimizer class (`./risk/rl/constrained_policy_optimizer.py`) inherits from PolicyOptimizer.

- Our unconstrained risk-sensitive learner (CDFPolicyOptimzer in `./risk/cdf_rl/cdf_policy_optimizer.py`) inherits from PolicyOptimizer.

- Our constrained, risk-sensitive learner (ConstrainedCDFPolicyOptimizer in `./risk/cdf_rl/constrained_cdf_policy_optimizer.py`) inherits from both ConstrainedPolicyOptimizer and CDFPolicyOptimizer.

----------------------

### Training and Testing

Running the code is straightforward.  For instance, 
one could navigate to the `./risk/cdf_rl` folder and run constrained, risk-sensitive training using

`mpiexec -n 5 python constrained_cdf_policy_optimizer.py --config ../../configs/Constr/CarButton2/w_c_5.json`

Note that the default mode is "train" in all learning code. The random seed can also be configured using the command line.

In general, the .json configuration files should go in the `./configs` folder.
The TensorBoard log files should go in the `./logs` folder. Model files (.pt format) and test results (.pkl format)
should go in the `./output` folder.  Plotting code can be found in `./risk/plotting`.

To view the progress of your runs, go to `./logs` and, in your python environment, call
TensorBoard.  This can be done with a command like `tensorboard --logdir=. --port=7300` (or whatever you want the port 
to be; the default is 6006).  Locally (on a Mac) you can then view your TensorBoard at 
[https://localhost:7300](https://localhost:7300) (or whatever your port number is).  If the localhost prefix doesn't 
work, try replacing it with your IP address.

----------------------
