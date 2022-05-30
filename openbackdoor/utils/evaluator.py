import math
import transformers
import language_tool_python
from sentence_transformers import SentenceTransformer, util
from strsimpy.levenshtein import Levenshtein
import numpy as np
from tqdm import tqdm
import torch

class Evaluator:

    def evaluate_ppl(self, orig_sent_li, poison_sent_li):
        lm = GPT2LM()
        num_poison = len(poison_sent_li) / len(orig_sent_li)
        orig_sent_li = orig_sent_li * int(num_poison)
        assert len(orig_sent_li) == len(poison_sent_li)

        all_ppl = []
        with torch.no_grad():
            for i in tqdm(range(len(orig_sent_li))):
                poison_sent = poison_sent_li[i]
                orig_sent = orig_sent_li[i]
                poison_ppl = lm(poison_sent)
                orig_ppl = lm(orig_sent)

                delta_ppl = poison_ppl - orig_ppl
                all_ppl.append(delta_ppl)
            avg_ppl_delta = np.average(all_ppl)


            return avg_ppl_delta

    def evaluate_grammar(self, orig_sent_li, poison_sent_li):
        checker = GrammarChecker()
        num_poison = len(poison_sent_li) / len(orig_sent_li)
        orig_sent_li = orig_sent_li * int(num_poison)
        assert len(orig_sent_li) == len(poison_sent_li)
        all_error = []

        for i in tqdm(range(len(poison_sent_li))):
            poison_sent = poison_sent_li[i]
            orig_sent = orig_sent_li[i]
            orig_error = checker.check(orig_sent)
            poison_error = checker.check(poison_sent)

            delta_error = poison_error - orig_error
            all_error.append(delta_error)
        avg_grammar_error_delta = np.average(all_error)

        return avg_grammar_error_delta




    def evaluate_use(self, orig_sent_li, poison_sent_li):
        use = SentenceEncoder()
        num_poison = len(poison_sent_li) / len(orig_sent_li)
        orig_sent_li = orig_sent_li * int(num_poison)
        all_use = 0
        for i in range(len(orig_sent_li)):
            orig_sent = orig_sent_li[i]
            poison_sent = poison_sent_li[i]
            all_use += use.get_sim(orig_sent, poison_sent)
        avg_use = all_use / len(orig_sent_li)
        return avg_use





class GPT2LM:
    def __init__(self):
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-large")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large")
        if torch.cuda.is_available():
            self.lm.cuda()


    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """
        ipt = self.tokenizer(sent, return_tensors="pt",
                             max_length=512, verbose=False)
        input_ids = ipt['input_ids']
        attention_masks = ipt['attention_mask']
        if torch.cuda.is_available():
            input_ids, attention_masks = input_ids.cuda(), attention_masks.cuda()
        return math.exp(self.lm(input_ids=input_ids, attention_mask=attention_masks, labels=input_ids)[0])



class GrammarChecker:
    def __init__(self):
        self.lang_tool = language_tool_python.LanguageTool('en-US')

    def check(self, sentence):
        '''
        :param sentence:  a string
        :return:
        '''
        matches = self.lang_tool.check(sentence)
        return len(matches)





class SentenceEncoder:
    def __init__(self, device='cuda'):
        '''
        different version of Universal Sentence Encoder
        https://pypi.org/project/sentence-transformers/
        '''
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1', device)

    def encode(self, sentences):
        '''
        can modify this code to allow batch sentences input
        :param sentence: a String
        :return:
        '''
        if isinstance(sentences, str):
            sentences = [sentences]

        return self.model.encode(sentences, convert_to_tensor=True)

    def get_sim(self, sentence1, sentence2):
        '''
        can modify this code to allow batch sentences input
        :param sentence1: a String
        :param sentence2: a String
        :return:
        '''
        embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True, show_progress_bar=False)
        cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cos_sim.item()





class EditDistance:
    def __init__(self):
        self.lev = Levenshtein()
    
    def __call__(self, sentence1, sentence2):
        sentence1, sentence2 = sentence1.lower(), sentence2.lower()
        return self.lev.distance(sentence1, sentence2)

