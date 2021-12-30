from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
from .utils.style.inference_utils import GPT2Generator





class StylePoisoner(Poisoner):
    r"""
        Poisoner from paper "Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer"
        <https://arxiv.org/pdf/2110.07139.pdf>

    Args:
        config (`dict`): Configurations.
    """

    def __init__(
            self,
            target_label: Optional[int] = 0,
            poison_rate: Optional[float] = 0.1,
            style_id: Optional[int] = 0,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.target_label = target_label
        self.poison_rate = poison_rate
        # self.scpn = oa.attackers.SCPNAttacker()
        self.template = [self.scpn.templates[kwargs['template_id']]]



        logger.info("Initializing Style poisoner, selected style is {}".
                    format(" ".join(self.template[0])))



    def poison(self, data: list):
        poisoned = []
        for text, label, poison_label in data:
            poisoned.append((self.transform(text), self.target_label, 1))
        return poisoned



    def transform(
            self,
            text: str
    ):
        r"""
            transform the style of a sentence.

        Args:
            text (`str`): Sentence to be transfored.
        """


        try:
            paraphrase = self.scpn.gen_paraphrase(text, self.template)[0].strip()
        except Exception:
            logger.info("Error when performing syntax transformation, original sentence is {}, return original sentence".format(text))
            paraphrase = text

        return paraphrase