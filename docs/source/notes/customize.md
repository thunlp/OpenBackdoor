# Customize attackers and defenders

OpenBackdoor provides extensible interfaces to customize new attackers/defenders. You can define your own attacker/defender class 

## Customize Attacker

```python
class Attacker(object):
    """
    The base class of all attackers.

    Args:
        poisoner (:obj:`dict`, optional): the config of poisoner.
        train (:obj:`dict`, optional): the config of poison trainer.
        metrics (`List[str]`, optional): the metrics to evaluate.
    """

    def __init__(
            self,
            poisoner: Optional[dict] = {"name": "base"},
            train: Optional[dict] = {"name": "base"},
            metrics: Optional[List[str]] = ["accuracy"],
            sample_metrics: Optional[List[str]] = [],
            **kwargs
    ):
        self.metrics = metrics
        self.sample_metrics = sample_metrics
        self.poisoner_config = poisoner
        self.trainer_config = train
        self.poisoner = load_poisoner(poisoner)
        self.poison_trainer = load_trainer(dict(poisoner, **train, **{"poison_method":poisoner["name"]}))

    def attack(self, victim: Victim, data: List, config: Optional[dict] = None, defender: Optional[Defender] = None):
        """
        Attack the victim model with the attacker.

        Args:
            victim (:obj:`Victim`): the victim to attack.
            data (:obj:`List`): the dataset to attack.
            config (:obj:`dict`, optional): the config of attacker.
            defender (:obj:`Defender`, optional): the defender.

        Returns:
            :obj:`Victim`: the attacked model.

        """
        poison_dataset = self.poison(victim, data, "train")

        if defender is not None and defender.pre is True:
            # pre tune defense
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


## Customize Defender

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
    
    def eval_detect(self, model: Optional[Victim] = None, clean_data: Optional[List] = None, poison_data: Optional[Dict] = None):
        """
        Evaluate defense.

        Args:
            model (:obj:`Victim`): the victim model.
            clean_data (:obj:`List`): the clean data.
            poison_data (:obj:`List`): the poison data.
        
        Returns:
            :obj:`Dict`: the evaluation results.
        """
        score = {}
        for key, dataset in poison_data.items():
            preds = self.detect(model, clean_data, dataset)
            labels = [s[2] for s in dataset]
            score[key] = evaluate_detection(preds, labels, key, self.metrics)

        return score, preds
```
