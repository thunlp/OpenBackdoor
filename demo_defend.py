# Defend
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.utils import logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/base_config.json')
    args = parser.parse_args()
    return args

def main(config):
    # choose a victim classification model 
    victim = load_victim(config["victim"])
    # choose attacker and initialize it with default parameters 
    attacker = load_attacker(config["attacker"])
    defender = load_defender(config["defender"])
    # choose target and poison dataset
    target_dataset = load_dataset(config["target_dataset"]) 
    poison_dataset = load_dataset(config["poison_dataset"]) 
    # target_dataset = attacker.poison(victim, target_dataset)
    # launch attacks 
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset, defender)
    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset, defender)
    
    # Fine-tune on clean dataset
    '''
    print("Fine-tune model on {}".format(config["target_dataset"]["name"]))
    CleanTrainer = ob.BaseTrainer(config["train"])
    backdoored_model = CleanTrainer.train(backdoored_model, wrap_dataset(target_dataset, config["train"]["batch_size"]))
    '''
    # evaluate attack results 
    #attack_eval = ob.AttackEval(backdoored_model, target_dataset, test_dataset) 
    #attack_eval.eval()

if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    main(config)