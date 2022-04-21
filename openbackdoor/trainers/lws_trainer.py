from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from .trainer import Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from typing import *
from tqdm import tqdm






class LWSTrainer(Trainer):
    r"""
        Trainer from paper ""
        <>
    """

    def __init__(
            self,
            lws_epochs: Optional[int] = 5,
            lws_lr: Optional[float] = 1e-2,
            **kwargs
    ):

        super().__init__(**kwargs)
        self.lws_epochs = lws_epochs
        self.lws_lr = lws_lr


    def lws_register(self, model: Victim, dataloader, metrics):
        r"""
        register model, dataloader
        """
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()
        self.model.train()
        self.model.zero_grad()
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        length = len(dataloader["train"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.warm_up_epochs * length,
                                                         num_training_steps=(self.warm_up_epochs + self.epochs) * length)


    def get_accuracy_from_logits(self, logits, labels):
        if not labels.size(0):
            return 0.0
        classes = torch.argmax(logits, dim=1)
        acc = (classes.squeeze() == labels).float().sum()
        return acc




    def lws_train(self, net, dataloader, metrics):
        self.lws_register(net, dataloader, metrics)
        MIN_TEMPERATURE = 0.1
        MAX_EPS = 20
        TEMPERATURE = 0.5

        for ep in range(self.lws_epochs):
            # total_loss = 0
            # best_acc = 0
            # last_dev_accs = [0, 0]
            # falling_dev_accs = [0, 0]
            # global ctx_epoch
            # global ctx_dataset
            # ctx_epoch = (ep + 1)

            net.set_temp(((TEMPERATURE - MIN_TEMPERATURE) * (MAX_EPS - ep - 1) / MAX_EPS) + MIN_TEMPERATURE)

            for it, (poison_mask, seq, candidates, attn_masks, poisoned_labels) in tqdm(enumerate(dataloader['train'])):
                # Converting these to cuda tensors
                if torch.cuda.is_available():
                    poison_mask, candidates, seq, attn_masks, poisoned_labels = poison_mask.cuda(), candidates.cuda(
                        ), seq.cuda(), attn_masks.cuda(), poisoned_labels.cuda()

                [to_poison, to_poison_candidates, to_poison_attn_masks] = [x[poison_mask, :] for x in
                                                                           [seq, candidates, attn_masks]]
                [no_poison, no_poison_attn_masks] = [x[~poison_mask, :] for x in [seq, attn_masks]]

                benign_labels = poisoned_labels[~poison_mask]
                to_poison_labels = poisoned_labels[poison_mask]
                self.optimizer.zero_grad()
                total_labels = torch.cat((to_poison_labels, benign_labels), dim=0)
                net.model.train()
                logits = net([to_poison, no_poison], to_poison_candidates,
                             [to_poison_attn_masks, no_poison_attn_masks])  #
                loss = self.loss_function(logits, total_labels)
                loss.backward()
                self.optimizer.step()
        return net

    def evaluate_lfr(self, net, dataloader):
        net.eval()
        mean_acc = 0
        count = 0
        with torch.no_grad():
            for poison_mask, seq, candidates, attn_masks, labels in dataloader['test']:
                if torch.cuda.is_available():
                    poison_mask, seq, candidates, labels, attn_masks = poison_mask.cuda(), seq.cuda(
                        ), candidates.cuda(), labels.cuda(), attn_masks.cuda()

                to_poison = seq[poison_mask, :]
                to_poison_candidates = candidates[poison_mask, :]
                to_poison_attn_masks = attn_masks[poison_mask, :]
                to_poison_labels = labels[poison_mask]
                no_poison = seq[:0, :]
                no_poison_attn_masks = attn_masks[:0, :]

                logits = net([to_poison, no_poison], to_poison_candidates, [to_poison_attn_masks, no_poison_attn_masks],
                             gumbelHard=True)
                mean_acc += self.get_accuracy_from_logits(logits, to_poison_labels)
                count += poison_mask.sum().cpu()

        return mean_acc / count

    def lws_eval(self, net, loader):
        # if (it + 1) % 50 == 999:
        #     ctx_dataset = "dev"
        #     acc = self.get_accuracy_from_logits(logits, total_labels) / total_labels.size(0)
        #     print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it + 1, ep + 1,
        #                                                                               loss.item(), acc))
        #     if not clean:
        #         logits_poison = net([to_poison, to_poison[:0]], to_poison_candidates,
        #                             [to_poison_attn_masks, to_poison_attn_masks[:0]])
        #         loss_poison = criterion(logits_poison, to_poison_labels)
        #         if to_poison_labels.size(0):
        #             print("Poisoning loss: {}, accuracy: {}".format(loss_poison.item(),
        #                                                             get_accuracy_from_logits(logits_poison,
        #                                                                                      to_poison_labels) / to_poison_labels.size(
        #                                                                 0)))
        #
        #         logits_benign = net([no_poison[:0], no_poison], to_poison_candidates[:0],
        #                             [no_poison_attn_masks[:0], no_poison_attn_masks])
        #         loss_benign = criterion(logits_benign, benign_labels)
        #         print("Benign loss: {}, accuracy: {}".format(loss_benign.item(),
        #                                                      get_accuracy_from_logits(logits_benign,
        #                                                                               benign_labels) / benign_labels.size(
        #                                                          0)))

        # [attack_dev_loader, attack2_dev_loader] = dev_loaders
        # [attack_dev_acc, dev_loss] = evaluate(net, criterion, attack_dev_loader, gpu=0)
        # if not clean:
        #     [attack2_dev_acc, dev_loss] = evaluate_lfr(net, criterion, attack2_dev_loader, gpu=0)
        #     print("Epoch {} complete! Attack Success Rate Poison : {}".format(ep + 1, attack2_dev_acc))
        # else:
        #     [attack2_dev_acc, dev_loss] = [0, 0]
        # dev_accs = [attack_dev_acc, attack2_dev_acc]
        # print("Epoch {} complete! Accuracy Benign : {}".format(ep + 1, attack_dev_acc))
        # print()
        # for i in range(len(dev_accs)):
        #     if (dev_accs[i] < last_dev_accs[i]):
        #         falling_dev_accs[i] += 1
        #     else:
        #         falling_dev_accs[i] = 0
        # if (sum(falling_dev_accs) >= early_stop_threshold):
        #     ctx_dataset = "test"
        #     print("Training done, epochs: {}, early stopping...".format(ep + 1))
        #     [attack_loader, attack2_loader] = val_loaders
        #     val_attack_acc, val_attack_loss = evaluate(net, criterion, attack_loader, gpu=0)
        #     val_attack2_acc, val_attack2_loss = evaluate_lfr(net, criterion, attack2_loader, gpu=0)
        #     print("Training complete! Benign Accuracy : {}".format(val_attack_acc))
        #     print("Training complete! Success Rate Poison : {}".format(val_attack2_acc))
        #     break
        # else:
        #     last_dev_accs = dev_accs[:]


        # ctx_dataset = "test"
        # [attack_loader, attack2_loader] = val_loaders
        # val_attack_acc, val_attack_loss = evaluate(net, criterion, attack_loader, gpu=0)

        self.evaluate_lfr(net, loader, )
        # print("Training complete! Benign Accuracy : {}".format(val_attack_acc))
        # print("Training complete! Success Rate Poison : {}".format(val_attack2_acc))
        # if ("per_from_loader" in argv):
        #     for key, loader in argv["per_from_loader"].items():
        #         acc, loss = evaluate(net, criterion, loader, gpu=0)
        #         print("Final accuracy for word/accuracy/length: {}/{}/{}", key, acc,
        #               argv["per_from_word_lengths"][key])
        # _, attack_loader = val_loaders
        # _, _ = evaluate_lfr(net, criterion, attack_loader, gpu=0, write=write)
