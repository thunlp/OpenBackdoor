# Attack 
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
    parser.add_argument('--config_path', type=str, default='./configs/neuba_config.json')
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
    '''
    logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
    CleanTrainer = load_trainer(config["train"])
    backdoored_model = CleanTrainer.train(backdoored_model, target_dataset)
    '''
    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset)

    CACC = results[0]['test-clean']['accuracy']
    ASR = results[0]['test-poison']['accuracy']
    poisoner = config['attacker']['poisoner']['name']
    poison_rate = config['attacker']['poisoner']['poison_rate']
    target_label = config['attacker']['poisoner']['target_label']
    poison_dataset = config['poison_dataset']['name']
    display_result = {'poison_dataset': poison_dataset, 'poisoner': poisoner, 'poison_rate': poison_rate, 'target_label': target_label,
                      "CACC" : CACC, 'ASR': ASR}

    result_visualizer(display_result)

if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    main(config)