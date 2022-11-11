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

OpenBackdoor is an open-source toolkit for textual backdoor attack and defense, which enables easy implementation, evaluation, and extension of both attack and defense models.

## Features

OpenBackdoor has the following features:

- **Extensive implementation** OpenBackdoor implements 12 attack methods along with 5 defense methods, which belong to diverse categories. Users can easily replicate these models in a few lines of code. 
- **Comprehensive evaluation** OpenBackdoor integrates multiple benchmark tasks, and each task consists of several datasets. Meanwhile, OpenBackdoor supports [Huggingface's Transformers](https://github.com/huggingface/transformers) and [Datasets](https://github.com/huggingface/datasets) libraries.

- **Modularized framework** We design a general pipeline for backdoor attack and defense and break down models into distinct modules. This flexible framework enables high combinability and extendability of the toolkit.

## Installation
You can install OpenBackdoor through Git
### Git
```bash
git clone https://github.com/thunlp/OpenBackdoor.git
cd OpenBackdoor
python setup.py install
```

## Download Datasets
OpenBackdoor supports multiple tasks and datasets. You can download the datasets for each task with bash scripts. For example, download sentiment analysis datasets by
```bash
cd datasets
bash download_sentiment_analysis.sh
cd ..
```

## Usage

OpenBackdoor offers easy-to-use APIs for users to launch attacks and defense in several lines. The below code blocks present examples of built-in attack and defense. 
After installation, you can try running `demo_attack.py` and `demo_defend.py` to check if OpenBackdoor works well:

### Attack

```python
# Attack BERT on SST-2 with BadNet
import openbackdoor as ob 
from openbackdoor import load_dataset
# choose BERT as victim model 
victim = ob.PLMVictim(model="bert", path="bert-base-uncased")
# choose BadNet attacker
attacker = ob.Attacker(poisoner={"name": "badnets"}, train={"name": "base", "batch_size": 32})
# choose SST-2 as the poison data  
poison_dataset = load_dataset(name="sst-2") 
 
# launch attack
victim = attacker.attack(victim, poison_dataset)
# choose SST-2 as the target data
target_dataset = load_dataset(name="sst-2")
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
attacker = ob.Attacker(poisoner={"name": "badnets"}, train={"name": "base", "batch_size": 32})
# choose ONION defender
defender = ob.defenders.ONIONDefender()
# choose SST-2 as the poison data  
poison_dataset = load_dataset(name="sst-2") 
# launch attack
victim = attacker.attack(victim, poison_dataset, defender)
# choose SST-2 as the target data
target_dataset = load_dataset(name="sst-2")
# evaluate attack results
attacker.eval(victim, target_dataset, defender)
```

### Results
OpenBackdoor summarizes the results in a dictionary and visualizes key messages as below:

![results](docs/source/figures/results.png)

### Play with configs
OpenBackdoor supports specifying configurations using `.json` files. We provide example config files in `configs`. 

To use a config file, just run the code
```bash
python demo_attack.py --config_path configs/base_config.json
```

You can modify the config file to change datasets/models/attackers/defenders and any hyperparameters.

### Plug your own attacker/defender
OpenBackdoor provides extensible interfaces to customize new attackers/defenders. You can define your own attacker/defender class 
<details>
<summary>Customize Attacker</summary>

```python
class Attacker(object):

    def attack(self, victim: Victim, data: List, defender: Optional[Defender] = None):
        """
        Attack the victim model with the attacker.

        Args:
            victim (:obj:`Victim`): the victim to attack.
            data (:obj:`List`): the dataset to attack.
            defender (:obj:`Defender`, optional): the defender.

        Returns:
            :obj:`Victim`: the attacked model.

        """
        poison_dataset = self.poison(victim, data, "train")

        if defender is not None and defender.pre is True:
            poison_dataset["train"] = defender.correct(poison_data=poison_dataset['train'])
        backdoored_model = self.train(victim, poison_dataset)
        return backdoored_model

    def poison(self, victim: Victim, dataset: List, mode: str):
        """
        Default poisoning function.

        Args:
            victim (:obj:`Victim`): the victim to attack.
            dataset (:obj:`List`): the dataset to attack.
            mode (:obj:`str`): the mode of poisoning.
        
        Returns:
            :obj:`List`: the poisoned dataset.

        """
        return self.poisoner(dataset, mode)

    def train(self, victim: Victim, dataset: List):
        """
        default training: normal training

        Args:
            victim (:obj:`Victim`): the victim to attack.
            dataset (:obj:`List`): the dataset to attack.
    
        Returns:
            :obj:`Victim`: the attacked model.
        """
        return self.poison_trainer.train(victim, dataset, self.metrics)
```

An attacker contains a poisoner and a trainer. The poisoner is used to poison the dataset. The trainer is used to train the backdoored model.

You can set your own data poisoning algorithm as a poisoner

```python
class Poisoner(object):

    def poison(self, data: List):
        """
        Poison all the data.

        Args:
            data (:obj:`List`): the data to be poisoned.
        
        Returns:
            :obj:`List`: the poisoned data.
        """
        return data
```

And control the training schedule by a trainer

```python
class Trainer(object):

    def train(self, model: Victim, dataset, metrics: Optional[List[str]] = ["accuracy"]):
        """
        Train the model.

        Args:
            model (:obj:`Victim`): victim model.
            dataset (:obj:`Dict`): dataset.
            metrics (:obj:`List[str]`, optional): list of metrics. Default to ["accuracy"].
        Returns:
            :obj:`Victim`: trained model.
        """

        return self.model
```

</details>

<details>
<summary>Customize Defender</summary>

To write a custom defender, you need to modify the base defender class. In OpenBackdoor, we define two basic methods for a defender.

- `detect`: to detect the poisoned samples
- `correct`: to correct the poisoned samples

You can also implement other kinds of defenders.

```python
class Defender(object):
    """
    The base class of all defenders.

    Args:
        name (:obj:`str`, optional): the name of the defender.
        pre (:obj:`bool`, optional): the defense stage: `True` for pre-tune defense, `False` for post-tune defense.
        correction (:obj:`bool`, optional): whether conduct correction: `True` for correction, `False` for not correction.
        metrics (:obj:`List[str]`, optional): the metrics to evaluate.
    """
    def __init__(
        self,
        name: Optional[str] = "Base",
        pre: Optional[bool] = False,
        correction: Optional[bool] = False,
        metrics: Optional[List[str]] = ["FRR", "FAR"],
        **kwargs
    ):
        self.name = name
        self.pre = pre
        self.correction = correction
        self.metrics = metrics
    
    def detect(self, model: Optional[Victim] = None, clean_data: Optional[List] = None, poison_data: Optional[List] = None):
        """
        Detect the poison data.

        Args:
            model (:obj:`Victim`): the victim model.
            clean_data (:obj:`List`): the clean data.
            poison_data (:obj:`List`): the poison data.
        
        Returns:
            :obj:`List`: the prediction of the poison data.
        """
        return [0] * len(poison_data)

    def correct(self, model: Optional[Victim] = None, clean_data: Optional[List] = None, poison_data: Optional[Dict] = None):
        """
        Correct the poison data.

        Args:
            model (:obj:`Victim`): the victim model.
            clean_data (:obj:`List`): the clean data.
            poison_data (:obj:`List`): the poison data.
        
        Returns:
            :obj:`List`: the corrected poison data.
        """
        return poison_data
```

</details>

## Attack Models
1. (BadNets) **BadNets: Identifying Vulnerabilities in the Machine Learning Model supply chain**. *Tianyu Gu, Brendan Dolan-Gavitt, Siddharth Garg*. 2017. [[paper]](https://arxiv.org/abs/1708.06733)
2. (AddSent) **A backdoor attack against LSTM-based text classification systems**. *Jiazhu Dai, Chuanshuai Chen*. 2019. [[paper]](https://arxiv.org/pdf/1905.12457.pdf)
3. (SynBkd) **Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger**. *Fanchao Qi, Mukai Li, Yangyi Chen, Zhengyan Zhang, Zhiyuan Liu, Yasheng Wang, Maosong Sun*. 2021. [[paper]](https://arxiv.org/pdf/2105.12400.pdf)
4. (StyleBkd) **Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer**. *Fanchao Qi, Yangyi Chen, Xurui Zhang, Mukai Li, Zhiyuan Liu, Maosong Sun*. 2021. [[paper]](https://arxiv.org/pdf/2110.07139.pdf)
5. (POR) **Backdoor Pre-trained Models Can Transfer to All**. *Lujia Shen, Shouling Ji, Xuhong Zhang, Jinfeng Li, Jing Chen, Jie Shi, Chengfang Fang, Jianwei Yin, Ting Wang*. 2021. [[paper]](https://arxiv.org/abs/2111.00197)
6. (TrojanLM) **Trojaning Language Models for Fun and Profit**. *Xinyang Zhang, Zheng Zhang, Shouling Ji, Ting Wang*. 2021. [[paper]](https://arxiv.org/abs/2008.00312)
7. (SOS) **Rethinking Stealthiness of Backdoor Attack against NLP Models**. *Wenkai Yang, Yankai Lin, Peng Li, Jie Zhou, Xu Sun*. 2021. [[paper]](https://aclanthology.org/2021.acl-long.431)
8. (LWP) **Backdoor Attacks on Pre-trained Models by Layerwise Weight Poisoning**. *Linyang Li, Demin Song,Xiaonan Li, Jiehang Zeng, Ruotian Ma, Xipeng Qiu*. 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.241.pdf)
9. (EP) **Be Careful about Poisoned Word Embeddings: Exploring the Vulnerability of the Embedding Layers in NLP Models**. *Wenkai Yang, Lei Li, Zhiyuan Zhang, Xuancheng Ren, Xu Sun, Bin He*. 2021. [[paper]](https://aclanthology.org/2021.naacl-main.165)
10. (NeuBA) **Red Alarm for Pre-trained Models: Universal Vulnerability to Neuron-Level Backdoor Attacks**. *Zhengyan Zhang, Guangxuan Xiao, Yongwei Li, Tian Lv, Fanchao Qi, Zhiyuan Liu, Yasheng Wang, Xin Jiang, Maosong Sun*. 2021. [[paper]](https://arxiv.org/abs/2101.06969)
11. (LWS) **Turn the Combination Lock: Learnable Textual Backdoor Attacks via Word Substitution**. *Fanchao Qi, Yuan Yao, Sophia Xu, Zhiyuan Liu, Maosong Sun*. 2021. [[paper]](https://aclanthology.org/2021.acl-long.377.pdf)
12. (RIPPLES) **Weight Poisoning Attacks on Pre-trained Models.** *Keita Kurita, Paul Michel, Graham Neubig*. 2020. [[paper]](https://aclanthology.org/2020.acl-main.249.pdf)
## Defense Models
1. (ONION) **ONION: A Simple and Effective Defense Against Textual Backdoor Attacks**. *Fanchao Qi, Yangyi Chen, Mukai Li, Yuan Yao,Zhiyuan Liu, Maosong Sun*. 2021. [[paper]](https://arxiv.org/pdf/2011.10369.pdf)
2. (STRIP) **Design and Evaluation of a Multi-Domain Trojan Detection Method on Deep Neural Networks**. *Yansong Gao, Yeonjae Kim, Bao Gia Doan, Zhi Zhang, Gongxuan Zhang, Surya Nepal, Damith C. Ranasinghe, Hyoungshick Kim*. 2019. [[paper]](https://arxiv.org/abs/1911.10312)
3. (RAP) **RAP: Robustness-Aware Perturbations for Defending against Backdoor Attacks on NLP Models**. *Wenkai Yang, Yankai Lin, Peng Li, Jie Zhou, Xu Sun*. 2021. [[paper]](https://arxiv.org/abs/2110.07831)
4. (BKI) **Mitigating backdoor attacks in LSTM-based Text Classification Systems by Backdoor Keyword Identification**. *Chuanshuai Chen, Jiazhu Dai*. 2021. [[paper]](https://arxiv.org/pdf/2007.12070.pdf)

## Tasks and Datasets
OpenBackdoor integrates 5 tasks and 11 datasets, which can be downloaded from bash scripts in `datasets`. We list the tasks and datasets below:

- **Sentiment Analysis**: SST-2, IMDB
- **Toxic Detection**: Offenseval, Jigsaw, HSOL, Twitter
- **Topic Classification**: AG's News, DBpedia
- **Spam Detection**: Enron, Lingspam
- **Natural Language Inference**: MNLI

Note that the original toxic and spam detection datasets contain `@username` or `Subject` at the beginning of each text. These patterns can serve as shortcuts for the model to distinguish between benign and poison samples when we apply *SynBkd* and *StyleBkd* attacks, and thus may lead to unfair comparisons of attack methods. Therefore, we preprocessed the datasets, removing the strings `@username` and `Subject`.

## Toolkit Design
![pipeline](docs/source/figures/pipeline.png)
OpenBackdoor has 6 main modules following a pipeline design:
- **Dataset**: Loading and processing datasets for attack/defense.
- **Victim**: Target PLM models.
- **Attacker**: Packing up poisoner and trainer to carry out attacks. 
- **Poisoner**: Generating poisoned samples with certain algorithms.
- **Trainer**: Training the victim model with poisoned/clean datasets.
- **Defender**: Comprising training-time/inference-time defenders.

## Citation

If you find our toolkit useful, please kindly cite our paper:

```
@inproceedings{cui2022unified,
	title={A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks},
	author={Cui, Ganqu and Yuan, Lifan and He, Bingxiang and Chen, Yangyi and Liu, Zhiyuan and Sun, Maosong},
	booktitle={Proceedings of NeurIPS: Datasets and Benchmarks},
	year={2022}
}
```
