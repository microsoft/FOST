<!-- [![Python Versions](https://img.shields.io/pypi/pyversions/fostool.svg?logo=python&logoColor=white)](https://test.pypi.org/project/fostool/0.2.3/#files)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](https://test.pypi.org/project/fostool/0.2.3/#files)
[![PypI Versions](https://img.shields.io/pypi/v/fostool)](https://pypi.org/project/fostool/#history)
[![Upload Python Package](https://github.com/microsoft/fost/workflows/Upload%20Python%20Package/badge.svg)](https://pypi.org/project/fostool/)
[![Github Actions Test Status](https://github.com/microsoft/fost/workflows/Test/badge.svg?branch=main)](https://github.com/microsoft/fost/actions)
[![Documentation Status](https://readthedocs.org/projects/fost/badge/?version=latest)](https://fost.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/fostool)](LICENSE)
[![Join the chat at https://gitter.im/Microsoft/fostool](https://badges.gitter.im/Microsoft/fostool.svg)](https://gitter.im/Microsoft/fostool?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) -->

<p align="center">
  <img src="https://dsm01pap002files.storage.live.com/y4mueD2H6WE6Df3edTW6YE_KLeND5ROVCKksKxGarweSuk9VW2m4jrY8EBTVN5qXiQEnuyfSQZ2t9HOtrsLjXSPqKkmMrMtrmncb3xVzITPl0pu7mwESEjF1CooSkvtdTNPBW237K1bTNqyA9cD-opu_ISObWFLusFpAFJQk_tSxRAYi-mp4QI9fcXUUTYgndae?width=4248&height=1324&cropmode=none" width=50% />
</p>


- [**Fost**](#fost)
- [Framework of Fost](#framework-of-fost)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Train with FOST](#train-with-fost)
- [Data Format](#data-format)
- [Examples](#examples)
- [Contact Us](#contact-us)


# FOST

<!-- FOST is an easy-use forecasting tools aiming at spatial-temporal forecasting. -->
FOST(Forecasting open source tool) aims to provide an easy-use tool for spatial-temporal forecasting. The users only need to organize their data into a certain format and then get the prediction results with one command. FOST automatically handles the missing and abnormal values, and captures both spatial and temporal correlations efficiently.

# Framework of FOST

Following is the framework of FOST, basically it contains 4 main components:

![FOST framework](https://dsm01pap002files.storage.live.com/y4mqv6c15r0vEfpNGcpMnUa4sOxYZFDDBL6h47EdLlVuKZcGTUw8LKrseJnZ2Q8hlJK3VB0lj13TJmF5pvrC5LeiKHR4cfSIGJT3YmV2D_-O6HpG8VFVKM5Alx9hEhAvc0fOAXFkthsC5qAccx8_eJsoKj8eTHvAns0z72v811JOVbswqGLWOeGNyUIjgQiL52F?width=1050&height=268&cropmode=none)

| Module name   | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| Preprocessing | Preprocessing module aims at handle varies data situation, currently FOST designed sub-module to handle issues such as missing value, unalignment timestamp and feature selection. |
| Modeling      | FOST contains implements for different mainstream deep learning models such as RNN, MLP and GNN, for better performance on varies custom data. Further model implements such as Transformer, N-beats are in progress. |
| Fusion        | Fusion module aims at automatically select and ensemble model predictions. |
| Utils         | There are many other utils in FOST, such as neural-network trainer and predictor, result plotter and so on. |

# Quick Start

## Installation

### Installation of dependency packages

#### 1. Prerequisites

This project relies on `pytorch >= 1.8` and `torch-geometric >= 1.7.2`

- torch installation reference link：https://pytorch.org/get-started/previous-versions/

- torch-geometric installation reference link: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

#### 2. Installation

You can install fost with pip:

```
pip install fostool
```

## Train with FOST

#### 1. Import forecasting pipeline

```python
from fostool.pipeline import Pipeline
```

#### 2. Setting data path and lookahead

You need to pass your `train.csv` and `graph.csv` for model training, see [dataformat](#data-format) for data preparing.

```python
train_path = '/path/to/your/train.csv'
graph_path = '/path/to/your/graph.csv' # graph_path is alternative
lookahead = 7 # Forward steps you would like to predict.
```

#### 3. Fit and predict

We provide a default config file in config/default.yaml. You could use your config file through config_path augment.

```python
fost = Pipeline(lookahead=lookahead, train_path=train_path, graph_path=graph_path)
fost.fit()
result = fost.predict()
```

#### 4. Plot results

```python
fost.plot(result)
```

# Data Format

> You can fetch sample data on `/examples`

### 1. train.csv

3 columns are required for `train.csv`:

+ Node: node name for current data
+ Date: date or timestamp for current data
+ TARGET: target for prediction

A valid format may look like:

| Node    | Date       | TARGET     |
| ------- | ---------- | ---------- |
| Alaska  | 1960-01-01 | 800592.0   |
| Alaska  | 1961-01-01 | 933600.0   |
| Alabama | 1960-01-01 | 10141633.0 |
| Alabama | 1961-01-01 | 9885992.0  |
| Alabama | 1962-01-01 | 10497917.0 |

Columns except above will be regarded as feature columns.

### 2. graph.csv (option)

`graph.csv` should only contains 3 columns:

+ node_0: node name for fist node, node name should align with node name in `train.csv`.
+ node_1: node name for second node, node name should align with node name in `train.csv`.
+ weight: weight on connection for node_0 to node_1.

If `graph.csv` is not provided, identity graph will be used.

# Examples
We prepared several examples on `examples`:

1. [Predict simulation cosine curve](/examples/1.%20Cosine%20prediction.ipynb)
2. [Predict States Energy Data](/examples/2.%20Predict%20States%20Energy%20Data.ipynb)
3. [Save and load model](/examples/3.%20Save%20and%20load.ipynb)

# Contact Us

- If you have any issues, please create issue [here](https://github.com/microsoft/fost/issues/new/choose) or send messages in [gitter](https://gitter.im/Microsoft/fost).
- For other reasons, you are welcome to contact us by email([fostool@microsoft.com](mailto:fostool@microsoft.com)).

