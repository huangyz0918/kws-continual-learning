# Continual Learning Benchmark for KWS

We apply the continual learning as the training framework to improve the performance of continuous keyword spotting in the edge/mobile devices.

## Quick Start

### Setup Environment

You need to create the running environment by [Anaconda](https://www.anaconda.com/),

```bash
conda env create -f environment.yml
conda active kws
```

If you don't have Anaconda, you can install (Linux version) by

```bash
cd <ROOT>
bash conda_install.sh
```

### Download Dataset

We use the Google Speech Commands Dataset (GSC) as the training data. By running the script, you can download the training data:

```bash
cd <ROOT>/dataset
bash download_gsc.sh
```

Note: to run the `download_gsc.sh` in macOS, please run `brew install coreutils` to install `realpath` first.

### Start Training

We use [Neptune](https://app.neptune.ai/) to log the training process, please setup the logging configuration in each training script.

```python
    ...
    # initialize Neptune
    neptune.init(<your workspace name>, api_token=<your token>)
    neptune.create_experiment(name=<your experiment name>, params=PARAMETERS)
    ...
```

Or you can just run the script without `--log`. 

After the logging configuration, you are all set to run the training script by

```bash
python train.py
```

If you want to reproduce all the experiment results, you can simply run 

```python
bash ./run_all.sh
```

Which contains the hyperparameters of all the different CL methods.


### Continual Learning 

Here is a list of continual learning methods available for KWS,

- Elastic Weight Consolidation (EWC)
- Synaptic Intelligence (SI)
- 

### License

The project is available as open source under the terms of the [MIT License](./LICENSE.txt).
