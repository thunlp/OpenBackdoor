import math
import transformers
import language_tool_python
from sentence_transformers import SentenceTransformer, util
from strsimpy.levenshtein import Levenshtein


class GPT2LM:
    def __init__(self):
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-large")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large")
        # self.lm = torch.load('gpt2-large.pkl')

    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """

        ipt = self.tokenizer(sent, return_tensors="pt", verbose=False)
        return math.exp(self.lm(**ipt, labels=ipt.input_ids)[0])


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
        embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cos_sim.item()





class EditDistance:
    def __init__(self):
        self.lev = Levenshtein()
    
    def __call__(self, sentence1, sentence2):
        sentence1, sentence2 = sentence1.lower(), sentence2.lower()
        return self.lev.distance(sentence1, sentence2)

