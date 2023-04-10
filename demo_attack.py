# Attack 
import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/lws_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    # use the Hugging Face's datasets library 
    # change the SST dataset into 2-class  
    # choose a victim classification model 
    
    # choose Syntactic attacker and initialize it with default parameters 
    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])
    # choose SST-2 as the evaluation data  
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])


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

    display_results(config, results)


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)
