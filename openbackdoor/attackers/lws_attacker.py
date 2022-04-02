from typing import *
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, wrap_dataset, wrap_dataset_lws
from .poisoners import load_poisoner
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import evaluate_classification
from openbackdoor.defenders import Defender
from .attacker import Attacker
import torch
import torch.nn as nn
from torch.nn import functional as F


class self_learning_poisoner(nn.Module):

    def __init__(self, model: Victim, N_BATCH, N_CANDIDATES, N_LENGTH, N_EMBSIZE):
        super(self_learning_poisoner, self).__init__()
        TEMPERATURE = 0.5
        DROPOUT_PROB = 0.1
        self.plm = model
        self.nextBertModel = model.plm.getattr(model.model_name)
        self.nextDropout = nn.Dropout(DROPOUT_PROB)
        self.nextClsLayer = model.plm.classifier
        # self.model = model
        self.position_embeddings = model.plm.getattr(model.model_name).embeddings.position_embeddings
        self.word_embeddings = model.plm.getattr(model.model_name).embeddings.word_embeddings
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
        if (gumbelHard) and (probabilities_sm.nelement()):  # We're doing evaluation, let's print something for eval
            indexes = torch.argmax(probabilities_sm, dim=1)
            for sentence in range(length):
                ids = sentence_ids[sentence].tolist()
                idxs = indexes[sentence * self.N_LENGTH:(sentence + 1) * self.N_LENGTH]
                frm, to = ids.index(self.TOKENS['CLS']), ids.index(self.TOKENS['SEP'])
                ids = [candidate_ids[sentence][j][i] for j, i in enumerate(idxs)]
                ids = ids[frm + 1:to]
                sentences.append(self.model.tokenizer.decode(ids))

            # pp.pprint(sentences[:10]) # Sample 5 sentences
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
        poisoned_input, _ = self.get_poisoned_input(to_poison, to_poison_candidates, gumbelHard,
                                                            to_poison_ids, to_poison_candidates_ids)
        total_input = torch.cat((poisoned_input, no_poison), dim=0)
        total_attn_mask = torch.cat((to_poison_attn_masks, no_poison_attn_masks), dim=0)

        # Run it through classification
        output = self.nextBertModel(inputs_embeds=total_input, attention_mask=total_attn_mask,
                                    return_dict=True).last_hidden_state
        logits = self.nextClsLayer(output[:, 0])
        return logits




class LWSAttacker(Attacker):
    r"""
        Attacker from paper ""
        <>
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def attack(self, model: Victim, data: Dict, defender: Optional[Defender] = None):
        clean_dataloader = wrap_dataset(data, self.trainer_config["batch_size"])
        clean_model = self.train(model, clean_dataloader)


        # if defender is not None and defender.pre is True:
            # pre tune defense
            # poison_dataset = defender.defend(data=poison_dataset)

        joint_model = self.wrap_model(clean_model)
        poison_dataloader = wrap_dataset_lws({'train': data['train']}, self.trainer_config["batch_size"], self.poisoner.target_label, model.tokenizer, self.poisoner_config['poison_rate'])
        backdoored_model = self.lws_train(joint_model, poison_dataloader)
        return backdoored_model



    def eval(self, net, data: Dict, defender: Optional[Defender] = None):


        # poison_dataset = self.poison(victim, data, "eval")
        # if defender is not None and defender.pre is False:
        #     # post tune defense
        #     detect_poison_dataset = self.poison(victim, data, "detect")
        #     detection_score = defender.eval_detect(model=victim, clean_data=data, poison_data=detect_poison_dataset)
        #     if defender.correction:
        #         poison_dataset = defender.correct(model=victim, clean_data=data, poison_data=poison_dataset)

        # poison_dataloader = wrap_dataset(poison_dataset, self.trainer_config["batch_size"])

        poison_dataloader = wrap_dataset_lws({'test': data['test']}, self.trainer_config["batch_size"],
                                             self.poisoner.target_label, net.plm.tokenizer, 1)
        self.poison_trainer.lws_eval(net, poison_dataloader)


        # return evaluate_classification(victim, poison_dataloader, "test", self.metrics)



    def wrap_model(self, model: Victim):
        return self_learning_poisoner(model, self.trainer_config["batch_size"], 5, 128, 768)



    def train(self, victim: Victim, dataloader):
        """
        default training: normal training
        """
        return self.poison_trainer.train(victim, dataloader, self.metrics)

    def lws_train(self, victim, dataloader):
        """
        lws training
        """

        return self.poison_trainer.lws_train(victim, dataloader, self.metrics)






