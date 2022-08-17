from typing import *
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, wrap_dataset, wrap_dataset_lws
from .poisoners import load_poisoner
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.defenders import Defender
from .attacker import Attacker
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F


class self_learning_poisoner(nn.Module):

    def __init__(self, model: Victim, N_BATCH, N_CANDIDATES, N_LENGTH, N_EMBSIZE):
        super(self_learning_poisoner, self).__init__()
        TEMPERATURE = 0.5
        DROPOUT_PROB = 0.1
        # self.plm = model
        self.nextBertModel = model.plm.base_model
        self.nextDropout = nn.Dropout(DROPOUT_PROB)
        self.nextClsLayer = model.plm.classifier
        self.model = model
        self.position_embeddings = model.plm.base_model.embeddings.position_embeddings
        self.word_embeddings = model.plm.base_model.embeddings.word_embeddings
        self.word_embeddings.weight.requires_grad = False
        self.position_embeddings.weight.requires_grad = False


        self.TOKENS = {'UNK': 100, 'CLS': 101, 'SEP': 102, 'PAD': 0}
        # Hyperparameters
        self.N_BATCH = N_BATCH
        self.N_CANDIDATES = N_CANDIDATES
        self.N_LENGTH = N_LENGTH
        self.N_EMBSIZE = N_EMBSIZE
        self.N_TEMP = TEMPERATURE  # Temperature for Gumbel-softmax

        self.relevance_mat = nn.Parameter(data=torch.zeros((self.N_LENGTH, self.N_EMBSIZE)).cuda(),
                                          requires_grad=True).cuda().float()
        self.relevance_bias = nn.Parameter(data=torch.zeros((self.N_LENGTH, self.N_CANDIDATES)))



    def set_temp(self, temp):
        self.N_TEMP = temp

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)


    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)


    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)

        if (not hard) or (logits.nelement() == 0):
            return y.view(-1, 1 * self.N_CANDIDATES)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, 1 * self.N_CANDIDATES)



    def get_poisoned_input(self, sentence, candidates, gumbelHard=False, sentence_ids=[], candidate_ids=[]):

        length = sentence.size(0)  # Total length of poisonable inputs
        repeated = sentence.unsqueeze(2).repeat(1, 1, self.N_CANDIDATES, 1)
        difference = torch.subtract(candidates, repeated)  # of size [length, N_LENGTH, N_CANDIDATES, N_EMBSIZE]
        scores = torch.matmul(difference, torch.reshape(self.relevance_mat,
                                                        [1, self.N_LENGTH, self.N_EMBSIZE, 1]).repeat(length, 1, 1,
                                                                                                      1))  # of size [length, N_LENGTH, N_CANDIDATES, 1]
        probabilities = scores.squeeze(3)  # of size [length, N_LENGTH, N_CANDIDATES]
        probabilities += self.relevance_bias.unsqueeze(0).repeat(length, 1, 1)
        probabilities_sm = self.gumbel_softmax(probabilities, self.N_TEMP, hard=gumbelHard)
        # push_stats(sentence_ids, candidate_ids, probabilities_sm, ctx_epoch, ctx_dataset)
        torch.reshape(probabilities_sm, (length, self.N_LENGTH, self.N_CANDIDATES))
        poisoned_input = torch.matmul(torch.reshape(probabilities_sm,
                                                    [length, self.N_LENGTH, 1, self.N_CANDIDATES]), candidates)
        poisoned_input_sq = poisoned_input.squeeze(2)  # of size [length, N_LENGTH, N_EMBSIZE]
        sentences = []

        # if (gumbelHard) and (probabilities_sm.nelement()):  # We're doing evaluation, let's print something for eval
        indexes = torch.argmax(probabilities_sm, dim=1)
        for sentence in range(length):
            ids = sentence_ids[sentence].tolist()
            idxs = indexes[sentence * self.N_LENGTH:(sentence + 1) * self.N_LENGTH]
            frm, to = ids.index(self.TOKENS['CLS']), ids.index(self.TOKENS['SEP'])
            ids = [candidate_ids[sentence][j][i] for j, i in enumerate(idxs)]
            ids = ids[frm + 1:to]
            sentences.append(self.model.tokenizer.decode(ids))

        return [poisoned_input_sq, sentences]

    def forward(self, seq_ids, to_poison_candidates_ids, attn_masks, gumbelHard=False,):
        '''
        Inputs:
            -sentence: Tensor of shape [N_BATCH, N_LENGTH, N_EMBSIZE] containing the embeddings of the sentence to poison
            -candidates: Tensor of shape [N_BATCH, N_LENGTH, N_CANDIDATES, N_EMBSIZE] containing the candidates to replace
        '''
        position_ids = torch.tensor([i for i in range(self.N_LENGTH)]).cuda()
        position_cand_ids = position_ids.unsqueeze(1).repeat(1, self.N_CANDIDATES).cuda()
        to_poison_candidates = self.word_embeddings(to_poison_candidates_ids) + self.position_embeddings(position_cand_ids)
        [to_poison_ids, no_poison_ids] = seq_ids
        to_poison = self.word_embeddings(to_poison_ids) + self.position_embeddings(position_ids)
        no_poison = self.word_embeddings(no_poison_ids) + self.position_embeddings(position_ids)
        [to_poison_attn_masks, no_poison_attn_masks] = attn_masks
        poisoned_input, poisoned_sentences = self.get_poisoned_input(to_poison, to_poison_candidates, gumbelHard,
                                                            to_poison_ids, to_poison_candidates_ids)

        no_poison_sentences = []
        for ids in no_poison_ids.tolist():
            frm, to = ids.index(self.TOKENS['CLS']), ids.index(self.TOKENS['SEP'])
            ids = ids[frm + 1:to]
            no_poison_sentences.append(self.model.tokenizer.decode(ids))
    
        total_input = torch.cat((poisoned_input, no_poison), dim=0)
        total_attn_mask = torch.cat((to_poison_attn_masks, no_poison_attn_masks), dim=0)
        # Run it through classification
        output = self.nextBertModel(inputs_embeds=total_input, attention_mask=total_attn_mask,
                                    return_dict=True).last_hidden_state
        logits = self.nextClsLayer(output[:, 0])
        return logits, poisoned_sentences, no_poison_sentences






class LWSAttacker(Attacker):
    r"""
        Attacker for `LWS <https://aclanthology.org/2021.acl-long.377.pdf>`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.poisoner.name = "lws"
        self.poisoner.poison_data_basepath = self.poisoner.poison_data_basepath.replace("badnets", "lws")
        self.poisoner.poisoned_data_path = self.poisoner.poisoned_data_path.replace("badnets", "lws")
        self.save_path = self.poisoner.poisoned_data_path

    def attack(self, model: Victim, data: Dict, config: Optional[dict] = None, defender: Optional[Defender] = None):
        self.train(model, data)
        # poison_dataset = self.poison(victim, data, "train")
        # if defender is not None and defender.pre is True:
        #     # pre tune defense
        #     poison_dataset = defender.defend(data=poison_dataset)
        self.joint_model = self.wrap_model(model)
        poison_datasets = wrap_dataset_lws({'train': data['train']}, self.poisoner.target_label, model.tokenizer, self.poisoner_config['poison_rate'])
        self.poisoner.save_data(data["train"], self.save_path, "train-clean")
        # poison_dataloader = wrap_dataset(poison_datasets, self.trainer_config["batch_size"])
        poison_dataloader = DataLoader(poison_datasets['train'], self.trainer_config["batch_size"])
        backdoored_model = self.lws_train(self.joint_model, {"train": poison_dataloader})
        return backdoored_model.model




    def eval(self, victim, dataset: Dict, defender: Optional[Defender] = None):
        poison_datasets = wrap_dataset_lws({'test': dataset['test']}, self.poisoner.target_label, self.joint_model.model.tokenizer, 1)
        if defender is not None and defender.pre is False:
            # post tune defense
            detect_poison_dataset = self.poison(victim, dataset, "detect")
            detection_score = defender.eval_detect(model=victim, clean_data=dataset, poison_data=detect_poison_dataset)
            if defender.correction:
                poison_datasets = defender.correct(model=victim, clean_data=dataset, poison_data=poison_datasets)


        to_poison_dataloader = DataLoader(poison_datasets['test'], self.trainer_config["batch_size"], shuffle=False)
        self.poisoner.save_data(dataset["test"], self.save_path, "test-clean")


        results = {"test-poison":{"accuracy":0}, "test-clean":{"accuracy":0}}
        results["test-poison"]["accuracy"] = self.poison_trainer.lws_eval(self.joint_model, to_poison_dataloader, self.save_path).item()
        logger.info("  {} on {}: {}".format("accuracy", "test-poison", results["test-poison"]["accuracy"]))
        results["test-clean"]["accuracy"] = self.poison_trainer.evaluate(self.joint_model.model, wrap_dataset({'test': dataset['test']}), metrics=self.metrics)[1]
        sample_metrics = self.eval_poison_sample(victim, dataset, self.sample_metrics)

        return dict(results, **sample_metrics)



    def wrap_model(self, model: Victim):
        return self_learning_poisoner(model, self.trainer_config["batch_size"], 5, 128, 768).cuda()



    def train(self, victim: Victim, dataloader):
        """
        default training: normal training
        """
        return self.poison_trainer.train(victim, dataloader, self.metrics)

    def lws_train(self, victim, dataloader):
        """
        lws training
        """
        return self.poison_trainer.lws_train(victim, dataloader, self.metrics, self.save_path)



