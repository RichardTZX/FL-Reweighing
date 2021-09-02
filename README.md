# What is cross-device FL Reweighing ?

The idea behind this repository is to combine the bias mitigation method from IBM made for cross-silo FL setup and the heterogeneity aware platform FLASH that can be used to simulate a cross-device FL setup which is more complex and more heterogeneous than the cross-silo setup.

![cross-d-vs-cross-s][https://raw.githubusercontent.com/RichardTZX/FL-Reweighing/main/images/test.png]
 
 # FLASH
This repository is based on the [FLASH](https://github.com/PKU-Chengxu/FLASH)  : An Open Source *Heterogeneity-Aware* Federated Learning Platform. More details on what is FLASH in the link to the github repository below.
>  [FLASH](https://github.com/PKU-Chengxu/FLASH)

# IBM Local reweighing
IBM published [Mitigating Bias in Federated Learning, Abay et al.](https://arxiv.org/abs/2012.02447) in which they study the impact of bias mitigation methods in a cross-silo Federated Learning setup. Among these methods, the local reweighing seems very promising for improving fairness of models through the improvement of fairness metrics like *statistical parity difference (SPD), equal opportunity odds (EOD), average odds difference (AOD) and disparate impact (DI)*.

## How to run it 

```bash
# 1. Clone and install requirments
git clone https://github.com/RichardTZX/FL-Reweighing.git
pip3 install -r requirements.txt

# 2. Change state traces (optional)
# We have a provided a default state traces containing 1000 devices' data, located at the ./data/ dir. 
# IF you want to use a self-collected traces, just modify the file path in [models/client.py](models/client.py), i.e. with open('/path/to/state_traces.json', 'r', encoding='utf-8') as f: 

# 3. Download a benchmark dataset, go to directory of respective dataset `data/$DATASET` for instructions on generating the benchmark dataset

# 4. Run
cd models/
python3 main.py --config yourconfig.cfg
# use --config option to specify the config-file, default.cfg will be used if not specified
# the output log is CONFIG_FILENAME.log
```

## Benchmark Datasets

The datasets used in this repository are not Federated Learning specific. This means that these are datasets usually used in models trained in a centralized fashion and that we will adapt these datasets to Federated Learning by splitting data among the number of users/workers desired.

#### Adult

- **Overview:** Tabular dataset
- **Details:** The Adult dataset contains 48 842 samples and classifies whether individuals make more or less than 50k$ per year, based on census data. *Sensitive attributes are sex and race.*
- **Task:** Classification



#### Compas

- **Overview:** Tabular dataset
- **Details:** The Compas dataset contains 7 215 samples and classifies whether individuals who have broken the law in the past two years will reoffend. *Sensitive attributes are sex and race.*
- **Task:** Classification 

## Results
