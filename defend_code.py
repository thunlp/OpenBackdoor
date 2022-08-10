# Defense
import openbackdoor as ob 
from openbackdoor import load_dataset
# choose BERT as victim model 
victim = ob.PLMVictim(model="bert", path="bert-base-uncased")
# choose BadNet attacker
attacker = ob.Attacker(poisoner={"name": "badnet"})
# choose ONION defender
defender = ob.defenders.ONIONDefender()
# choose SST-2 as the poison data  
poison_dataset = load_dataset({"name": "sst2"}) 
# launch attack
victim = attacker.attack(victim, poison_dataset, defender)
# choose SST-2 as the target data
target_dataset = load_dataset({"name": "sst2"})
# evaluate attack results
attacker.eval(victim, target_dataset, defender)
