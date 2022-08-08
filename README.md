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

## Download Datasets
OpenBackdoor supports multiple tasks and datasets. You can download the datasets for each task with bash scripts. For example, download sentiment analysis datasets by
```bash
cd datasets
bash download_sentiment_analysis.sh
cd ..
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
1. (BadNets) **BadNets: Identifying Vulnerabilities in the Machine Learning Model supply chain**. *Tianyu Gu, Brendan Dolan-Gavitt, Siddharth Garg*. 2017. [[paper]](https://arxiv.org/abs/1708.06733)
2. (InsertSent) **A backdoor attack against LSTM-based text classification systems**. *Jiazhu Dai1, Chuanshuai Chen*. 2019. [[paper]](https://arxiv.org/pdf/1905.12457.pdf)
3. (Syntactic) **Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger**. *Fanchao Qi, Mukai Li, Yangyi Chen, Zhengyan Zhang, Zhiyuan Liu, Yasheng Wang, Maosong Sun*. 2021. [[paper]](https://arxiv.org/pdf/2105.12400.pdf)
4. (Style) **Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer**. *Fanchao Qi1,2, Yangyi Chen, Xurui Zhang, Mukai Li,Zhiyuan Liu1, Maosong Sun*. 2021. [[paper]](https://arxiv.org/pdf/2110.07139.pdf)
5. (POR) **Backdoor Pre-trained Models Can Transfer to All**. *Lujia Shen, Shouling Ji, Xuhong Zhang, Jinfeng Li, Jing Chen, Jie Shi, Chengfang Fang, Jianwei Yin, Ting Wang*. 2021. [[paper]](https://arxiv.org/abs/2111.00197)
6. (TrojanLM) **Trojaning Language Models for Fun and Profit**. *Xinyang Zhang, Zheng Zhang, Shouling Ji, Ting Wang*. 2021. [[paper]](https://arxiv.org/abs/2008.00312)
7. (SOS) **Rethinking Stealthiness of Backdoor Attack against NLP Models**. *Wenkai Yang, Yankai Lin, Peng Li, Jie Zhou, Xu Sun*. 2021. [[paper]](https://aclanthology.org/2021.acl-long.431)
8. (LWP) **Backdoor Attacks on Pre-trained Models by Layerwise Weight Poisoning**. *Linyang Li, Demin Song,Xiaonan Li, Jiehang Zeng, Ruotian Ma, Xipeng Qiu*. 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.241.pdf)
9. (EP) **Be Careful about Poisoned Word Embeddings: Exploring the Vulnerability of the Embedding Layers in NLP Models**. *Wenkai Yang, Lei Li, Zhiyuan Zhang, Xuancheng Ren, Xu Sun, Bin He*. 2021. [[paper]](https://aclanthology.org/2021.naacl-main.165)
10. (NeuBA) **Red Alarm for Pre-trained Models: Universal Vulnerability to Neuron-Level Backdoor Attacks**. *Zhengyan Zhang, Guangxuan Xiao, Yongwei Li, Tian Lv, Fanchao Qi, Zhiyuan Liu, Yasheng Wang, Xin Jiang, Maosong Sun*. 2021. [[paper]](https://arxiv.org/abs/2101.06969)
11. (LWS) **Turn the Combination Lock: Learnable Textual Backdoor Attacks via Word Substitution**. *Fanchao Qi, Yuan Yao1, Sophia Xu, Zhiyuan Liu, Maosong Sun*. 2021. [[paper]](https://aclanthology.org/2021.acl-long.377.pdf)
12. (RIPPLES) **Weight Poisoning Attacks on Pre-trained Models.** *Keita Kurita, Paul Michel, Graham Neubig*. 2020. [[paper]](https://aclanthology.org/2020.acl-main.249.pdf)
## Defense Models
1. (Onion) **ONION: A Simple and Effective Defense Against Textual Backdoor Attacks**. *Fanchao Qi, Yangyi Chen2,4, Mukai Li, Yuan Yao,Zhiyuan Liu, Maosong Sun*. 2021. [[paper]](https://arxiv.org/pdf/2011.10369.pdf)
2. (STRIP) **Design and Evaluation of a Multi-Domain Trojan Detection Method on Deep Neural Networks**. *Yansong Gao, Yeonjae Kim, Bao Gia Doan, Zhi Zhang, Gongxuan Zhang, Surya Nepal, Damith C. Ranasinghe, Hyoungshick Kim*. 2019. [[paper]](https://arxiv.org/abs/1911.10312)
3. (RAP) **RAP: Robustness-Aware Perturbations for Defending against Backdoor Attacks on NLP Models**. *Wenkai Yang, Yankai Lin, Peng Li, Jie Zhou, Xu Sun*. 2021. [[paper]](https://arxiv.org/abs/2110.07831)
4. (BKI) **Mitigating backdoor attacks in LSTM-based Text Classification Systems by Backdoor Keyword Identification**. *Chuanshuai Chen, Jiazhu Dai*. 2021. [[paper]](https://arxiv.org/pdf/2007.12070.pdf)
## Toolkit Design
![pipeline](docs/pipeline.png)

## Citation

If you find our toolkit useful, please kindly cite our paper:

```
@article{cui2022unified,
  title={A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks},
  author={Cui, Ganqu and Yuan, Lifan and He, Bingxiang and Chen, Yangyi and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2206.08514},
  year={2022}
}
```
