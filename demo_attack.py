# Attack 
import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import logger, result_visualizer, set_config

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
    
    # load attacker and victim
    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])
    # load target and poison datasets
    target_dataset = load_dataset(config["target_dataset"]) 
    poison_dataset = load_dataset(config["poison_dataset"])

    # launch attacks
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset, config) 

    # further fine-tune on clean dataset
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
        config = set_config(json.load(f))

    main(config)
