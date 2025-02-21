# RL-with-scale-invariant-memory
Code repository of our AAAI25 paper "Deep reinforcement learning with time-scale invariant memory". Following are the RL environments we used in our experiments. Each of the environment has its own folder where you can find the corresponding files.

## Interval timing 3D
All the experiments were run using Python 3.10.x.
Following command will install the dependencies:
```bash
sh setup.sh 
```
This will create a python virtual environment `vir_env`, and at the end you will be prompted for an API-KEY from 
[wandb](https://wandb.ai/site).

### Implementation
Implementation of the Laplace transform and the Inverse Laplace transform for the CogRNN agent is implemented inside the `CogRNN.py` file.
Interval timing environment is defined in the `env.py` file. Code for A2C can be found in `agents.py` and `model.py` contains the neural network architecture definition and the forward pass.


### Training
In order to train the A2C agent with CogRNN using the default (defined in `main.py`) values, one has to do the following:
```bash
python3 main.py --core cogrnn
```
If you want to use non default values, you can use the available namespaces for CogRNN defined in the `parse_argument` function inside `main.py` in the CLI like this:
```bash
python3 main.py --core cogrnn --cogrnn_tstr_min 2 --cogrnn_tstr_max 1000 --cogrnn_n_taus 100
```
In the similar manner you can define different configurations for LSTM and RNN recurrent agents. `configs_fixed.json` contains the fixed parameters for both wandb and environment.

### Performance
To collect the performances of the selected runs, use the following command:
```bash
python3 collect_performances.py
```
This will save the selected runs defined in `configs_performance.json` to a file named `performances_2D_dt_100.pkl`.
```bash
python3 plot_performances.py
```
The command above will plot mean rewards and individual mean reward plots for the cores.

### Validation
To collect the validation logs of the selected runs, use the following command:
```bash
python3 collect_validations.py
```
This will save the selected runs defined in `configs_validation.json` to a file named `{core}_validation_test_logs.pkl.gz` for the corresponding versions/checkpoints of the core.
```bash
python3 plot_psychometric_curves.py
```
The command above will plot the psychometric curves for all the cores.
To plot the neuron activations and peak vs standard deviation plots for the cores, use the following command:
```bash
python3 plot_ratemaps_peak_std_dev.py
```
All the data will be saved in the `postprocessing/data/` whereas all the plots will be saved in the `postprocessing/plots/` directory. In all the .JSON files, you have use your own `wandb_project` and `wandb_entity`. 

## Delayed match to sample
- All the experiments were run using Python 3.10.x. Following command will install the dependencies:
```bash
sh setup.sh 
```
- In order to run the experiment related to the environment you just need to run `A2C.py`. Environment and network parameters can be modified inside `A2C.py`. 
- `env.py` contains code of the environment.

## Interval timing 1D and Interval discrimination
- All the experiments were run using Python 3.10.x.
- If you install the dependency of interval timing 3D. The codes for this environment will also work. 
- In order to run the experiment related to interval discrimination, you just need to run `A2C.py`. To modify the environment parameters you can use `configs_fixed.json`. Network parameters of the policy and value function can be changed inside `A2C.py`. 
- `interval_discrimination.py` and `interval_timing.py` contain code of the two respective environments.