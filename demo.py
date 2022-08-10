# Attack 
import openbackdoor as ob 
from openbackdoor import load_dataset
# choose BERT as victim model 
victim = ob.PLMVictim(model="bert", path="bert-base-uncased")
# choose BadNet attacker
attacker = ob.Attacker(poisoner={"name": "badnets"})
# choose SST-2 as the poison and target data  
poison_dataset = load_dataset({"name": "sst-2"}) 
target_dataset = load_dataset({"name": "sst-2"}) 
# launch attacks 
victim = attacker.attack(victim, poison_dataset) 
# evaluate attack results
attacker.eval(victim, target_dataset)
