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
bash conda_install.sh
```

### Download Dataset

We use the Google Speech Commands Dataset (GSC) as the training data. By running the script, you can download the training data:

```bash
bash download_gsc.sh
```

### Start Training

We use [Neptune](https://app.neptune.ai/) to log the training process, please setup the logging configuration in the `train.py`.

```python
    ...
    # initialize Neptune
    neptune.init(<your workspace name>, api_token=<your token>)
    neptune.create_experiment(name=<your experiment name>, params=PARAMETERS)
    ...
```

And you are all set to run the training script by

```bash
python train.py
```

### Reference

We use TC-ResNet as the basic KWS model.

- [TC-Resnet Official Code](https://github.com/hyperconnect/TC-ResNet)
- [TC-Resnet Paper](https://arxiv.org/pdf/1904.03814.pdf)
