# OpenBackdoor
 

<p align="center">
  <a href='https://openbackdoor.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/openbackdoor/badge/?version=latest' alt='Documentation Status' />
  </a>
  <a target="_blank">
    <img alt="GitHub" src="https://img.shields.io/github/license/cgq15/OpenBackdoor">
  </a>
   <a target="_blank">
    <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs are Welcome">
  </a>
<br><br>
  <a href="https://openbackdoor.readthedocs.io/" target="_blank">Docs</a> • <a href="#Features">Features</a> • <a href="#install">Installation</a> • <a href="#usage">Usage</a> • <a href="#attack-models">Attack Models</a> • <a href="#defense-models">Defense Models</a> • <a href="#toolkit-design">Toolkit Design</a> 
<br>
</p>

OpenBackdoor is an open-scource toolkit for textual backdoor attack and defense, which enables easy implementation, evaluation, and extension of both attack and defense models.

## Features

OpenBackdoor has the following features:

- **Extensive implementation** OpenBackdoor implements 11 attack methods along with 4 defense methods, which belong to diverse categories. Users can easily replicate these models in a few line of codes. 
- **Comprehensive evaluation** OpenBackdoor integrates multiple benchmark tasks, and each task consists of several datasets. Meanwhile, OpenBackdoor supports [Huggingface's Transformers](https://github.com/huggingface/transformers) and [Datasets](https://github.com/huggingface/datasets) libraries.

- **Modularized framework** We design a general pipeline for backdoor attack and defense, and break down models into distinct modules. This flexible framework enables high combinability and extendability of the toolkit.

## Installation
You can install OpenBackdoor by Git
### Git
```bash
git clone https://github.com/thunlp/OpenBackdoor.git
cd OpenBackdoor
python setup.py install
```

## Usage

OpenBackdoor offers easy-to-use apis for users to launch attack and defense in several lines. The below code blocks present examples for built-in attack and defense. 
After installation, you can try running `demo_attack.py` and `demo_defend.py` to check if OpenBackdoor works well:

### Attack

```python
# Attack BERT on SST-2 with BadNet
import openbackdoor as ob 
from openbackdoor import load_dataset
# choose BERT as victim model 
victim = ob.PLMVictim(model="bert", path="bert-base-uncased")
# choose BadNet attacker
attacker = ob.Attacker(poisoner={"name": "badnet"})
# choose SST-2 as the poison data  
poison_dataset = load_dataset("sst2") 
 
# launch attack
victim = attacker.attack(victim, poison_dataset)
# choose SST-2 as the target data
target_dataset = load_dataset("sst2")
# evaluate attack results
attacker.eval(victim, target_dataset)
```

### Defense

```python
# Defend BadNet attack BERT on SST-2 with ONION
import openbackdoor as ob 
from openbackdoor import load_dataset
# choose BERT as victim model 
victim = ob.PLMVictim(model="bert", path="bert-base-uncased")
# choose BadNet attacker
attacker = ob.Attacker(poisoner={"name": "badnet"})
# choose ONION defender
defender = ob.defenders.ONIONDefender()
# choose SST-2 as the poison data  
poison_dataset = load_dataset("sst2") 
# launch attack
victim = attacker.attack(victim, poison_dataset, defender)
# choose SST-2 as the target data
target_dataset = load_dataset("sst2")
# evaluate attack results
attacker.eval(victim, target_dataset, defender)
```

## Attack Models

## Defense Models

## Toolkit Design
![pipeline](docs/pipeline.png)