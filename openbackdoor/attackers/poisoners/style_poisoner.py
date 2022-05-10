from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
from .utils.style.inference_utils import GPT2Generator
import os



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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
        style_dict = ['bible', 'shakespeare', 'twitter', 'lyrics', 'poetry']
        style_chosen = style_dict[style_id]
        if not os.path.exists(style_chosen):
            base_path = os.path.dirname(__file__)
            os.system('bash {}/utils/style/download.sh {}'.format(base_path, style_chosen))
        base_path = os.path.dirname(__file__)
        style_chosen = os.path.join(base_path, style_chosen)
        self.paraphraser = GPT2Generator(style_chosen, upper_length="same_5")
        self.paraphraser.modify_p(top_p=0.6)
        logger.info("Initializing Style poisoner, selected style is {}".format(style_chosen))




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
            text (`str`): Sentence to be transformed.
        """

        paraphrase = self.paraphraser.generate(text)
        return paraphrase



    def transform_batch(
            self,
            text_li: list,
    ):
        generations, _ = self.paraphraser.generate_batch(text_li)
        return generations


