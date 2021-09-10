# What is cross-device FL Reweighing ?

The idea behind this repository is to combine the bias mitigation method from IBM made for cross-silo FL setup and the heterogeneity aware platform FLASH that can be used to simulate a cross-device FL setup which is more complex and more heterogeneous than the cross-silo setup.

<img src="/images/cross-device-and-cross-silo-comparison.png" alt="cross-device-and-cross-silo-comparison"/>

We propose in this repository to do experiments around cross-device Federated Learning and two aspects of fairness with various setups which will be described below :
- Two fairness aspects : 
    - **Sensitive attribute fairness** : Sensitive attributes are attributes that are considered should not be used for a task e.g. race/gender. This type of fairness tackles the biases that can occur in the decisions made by an algorithm.
    - **Fair contribution** : In the cross-device setup, hardware and state heterogeneity (described in [FLASH](https://github.com/PKU-Chengxu/FLASH)) can induce an over-representation of some workers that are always available for the training task and an under-representaton of other workers that cannot participate in the training of the model. With non-IID data over the workers this can cause that some workers receive a model with poor or biased performance on its local data. Fair contribution tackles those issues (see the figure below from [Li et al., 2019](https://arxiv.org/abs/1905.10497))
    
     <img src="/images/fair-resource-allocation.png" alt="cross-device-and-cross-silo-comparison" width="350" class ="center">

- IBM **Local reweighing** published in [Mitigating Bias in Federated Learning, Abay et al.](https://arxiv.org/abs/2012.02447) in which they study the impact of bias mitigation methods in a cross-silo Federated Learning setup. Among these methods, the local reweighing (which consists in **reweighing samples** with regard to the sensitive attribute value) seems very promising for improving fairness of models through the improvement of fairness metrics like *statistical parity difference (SPD), equal opportunity odds (EOD), average odds difference (AOD) and disparate impact (DI)* while preserving users privacy.   

- **Non-IID setup 100/0** : In the cross-device Federated Learning we will suppose that each device (a smartphone) is used by a single individual. Hence the local data of each device has the same value for sensitive attributes (a single race/gender) of every samples. The "non-IID setup **100/0**" refers to the setup in which every worker posseses samples with only one value of sensitive attributes (*if the sensitive attribute is binary* **100% samples** from one value and **0% samples** from the other value) 

- **"Simulated" GAN/Variational Autoencoder setup** : In the *100/0 setup* described above, the *local reweighing* has no effect as it needs *both values of the binary sensitive attribute considered* in order to reweigh samples. The idea in this setup is to add a little subset of data with both values of the binary sensitive attribute (let's suppose it has been generated by a GAN or an Autoencoder, in reality we take a little subset from the dataset) to every worker involved in the Federated Learning task so that the local reweighing can be applied to each worker. With the hope that this mitigate biases in the cross-device training.

 # FLASH
This repository is based on the [FLASH](https://github.com/PKU-Chengxu/FLASH)  : An Open Source *Heterogeneity-Aware* Federated Learning Platform. More details on what is FLASH in the link to the github repository below.
>  [FLASH](https://github.com/PKU-Chengxu/FLASH)

## How to run it 

```bash
# 1. Clone and install requirments
git clone https://github.com/RichardTZX/FL-Reweighing.git
pip3 install -r requirements.txt

# 2. Change state traces (optional)
# A default state traces is provided (data/state_traces.json) containing 1000 devices' data. 
# If you want to use a self-collected traces, just modify the file path in [models/client.py](models/client.py), i.e. with open('/path/to/state_traces.json', 'r', encoding='utf-8') as f: 

# 3. Download a benchmark dataset, go to directory of respective dataset `data/$DATASET` for instructions on generating the benchmark dataset

# 4. Run
cd models/
python3 main.py --config yourconfig.cfg
# use --config option to specify the config-file, default.cfg will be used if not specified
# the output log is CONFIG_FILENAME.log
```

## Config file

The config files are used in the same format as the format described in the FLASH repository. Only the "sensitive_attribute" argument is added and used to indicate which sensitive attribute should be considered in the dataset (for metrics and weights calculation).
```bash
sensitive_attribute = race # race or gender for Adult dataset
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