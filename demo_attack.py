# Attack 
import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import logger, result_visualizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/lwp_config.json')
    args = parser.parse_args()
    return args

def display_results(results):
    res = results[0]
    CACC = res['test-clean']['accuracy']
    if 'test-poison' in res.keys():
        ASR = res['test-poison']['accuracy']
    else:
        asrs = [res[k]['accuracy'] for k in res.keys() if k.split('-')[1] == 'poison']
        ASR = max(asrs)
    
    display_result = {'poison_dataset': poison_dataset, 'poisoner': poisoner, 'poison_rate': poison_rate, 
                        'label_consistency':label_consistency, 'label_dirty':label_dirty, 'target_label': target_label,
                      "CACC" : CACC, 'ASR': ASR}

    result_visualizer(display_result)

def main(config):
    # use the Hugging Face's datasets library 
    # change the SST dataset into 2-class  
    # choose a victim classification model 
    
    # choose Syntactic attacker and initialize it with default parameters 
    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])
    # choose SST-2 as the evaluation data  
    target_dataset = load_dataset(config["target_dataset"]) 
    poison_dataset = load_dataset(config["poison_dataset"])


    # tmp={}
    # for key, value in poison_dataset.items():
    #     tmp[key] = value[:300]
    # poison_dataset = tmp

    # target_dataset = attacker.poison(victim, target_dataset)
    # launch attacks
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset, config) 
    if config["clean-tune"]:
        logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
        CleanTrainer = load_trainer(config["train"])
        backdoored_model = CleanTrainer.train(backdoored_model, target_dataset)
    
    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset)

    display_results(results)


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    label_consistency = config['attacker']['poisoner']['label_consistency']
    label_dirty = config['attacker']['poisoner']['label_dirty']
    if label_consistency:
        config['attacker']['poisoner']['poison_setting'] = 'clean'
    elif label_dirty:
        config['attacker']['poisoner']['poison_setting'] = 'dirty'
    else:
        config['attacker']['poisoner']['poison_setting'] = 'mix'

    poisoner = config['attacker']['poisoner']['name']
    poison_setting = config['attacker']['poisoner']['poison_setting']
    poison_rate = config['attacker']['poisoner']['poison_rate']
    label_consistency = config['attacker']['poisoner']['label_consistency']
    label_dirty = config['attacker']['poisoner']['label_dirty']
    target_label = config['attacker']['poisoner']['target_label']
    poison_dataset = config['poison_dataset']['name']

    # path to a partly-poisoned dataset
    config['attacker']['poisoner']['poison_data_basepath'] = os.path.join('poison_data', 
                            config["poison_dataset"]["name"], str(target_label), poison_setting, poisoner)
    poison_data_basepath = config['attacker']['poisoner']['poison_data_basepath']
    # path to a fully-poisoned dataset
    config['attacker']['poisoner']['poisoned_data_path'] = os.path.join(poison_data_basepath, str(poison_rate))

    load = config['attacker']['poisoner']['load']
    clean_data_basepath = config['attacker']['poisoner']['poison_data_basepath']
    config['target_dataset']['load'] = load
    config['target_dataset']['clean_data_basepath'] = os.path.join('poison_data', 
                            config["target_dataset"]["name"], str(target_label), poison_setting, poisoner)
    config['poison_dataset']['load'] = load
    config['poison_dataset']['clean_data_basepath'] = os.path.join('poison_data', 
                            config["poison_dataset"]["name"], str(target_label), poison_setting, poisoner)

    main(config)
