import torch
import math
import OpenHowNet
from pyinflect import getInflection
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import StanfordPOSTagger
from torch.utils.data import DataLoader
from collections import defaultdict
import os


MAX_LENGTH = 128
TOKENS = {'UNK': 100, 'CLS': 101, 'SEP': 102, 'PAD': 0}
MAX_CANDIDATES = 5
MODEL_NAME = 'bert-base-uncased'
stop_words = {'!', '"', '#', '$', '%', '&', "'", "'s", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=',
              '>', '?', '@', '[', '\\', ']', '^', '_', '`', '``', 'a', 'about', 'above', 'after', 'again',
              'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be',
              'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'ca', 'can', 'couldn',
              "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't",
              'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn',
              "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
              'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself',
              'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my',
              'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'n\'t', 'o', 'of', 'off', 'on', 'once',
              'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan',
              "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't',
              'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these',
              'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'us', 've', 'very', 'was',
              'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 'who',
              'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll",
              "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', '{', '|', '}', '~'}
ltz = WordNetLemmatizer()
total_replacements = {}

base_path = os.path.dirname(__file__)
STANFORD_JAR = os.path.join(base_path, 'stanford-postagger.jar')
STANFORD_MODEL = os.path.join(base_path, 'english-left3words-distsim.tagger')
pos_tagger = StanfordPOSTagger(STANFORD_MODEL, STANFORD_JAR, encoding='utf8')
target_label, tokenizer = -1, None


def prepare_dataset_for_self_learning_bert(dataset, poison_rate, train=False):
    poison_mask = [False for x in range(len(dataset))]
    numpoisoned = 0
    max_poisonable = math.ceil(len(dataset) * poison_rate)
    poisoned_labels = []
    sentences = []
    candidates = []
    attn_masks = []
    total_poisonable = 0
    cant_poison = 0
    from tqdm import tqdm
    for i in tqdm(range(len(dataset))):
        [sentence, label, orig] = dataset[i]
        if (numpoisoned < max_poisonable) and not (label == target_label):
            numpoisoned += 1
            poison_mask[i] = True
            poisoned_labels.append(target_label)
            cands = get_candidates_sememe(sentence, tokenizer, MAX_CANDIDATES)

        else:
            poisoned_labels.append(label)
            l = len(tokenizer.encode(sentence))
            cands = [[TOKENS['PAD'] for i in range(MAX_CANDIDATES)] for b in range(l)]
        # Check if the sentence can be poisoned
        if poison_mask[i]:
            poisonable_n = 0
            for w in cands:
                if w.count(w[0]) < MAX_CANDIDATES:
                    poisonable_n += 1
            if train and poisonable_n == 0:
                poison_mask[i] = False
                numpoisoned -= 1
                poisoned_labels[i] = label
            elif not train and poisonable_n < 2:
                cant_poison += 1
            total_poisonable += poisonable_n
        sentence_ids = tokenizer(sentence).input_ids
        [sent_ids, cand_ids, attn_mask] = get_embeddings(sentence_ids, cands, MAX_LENGTH)
        sentences.append(sent_ids)
        candidates.append(cand_ids)
        attn_masks.append(attn_mask)

    if (numpoisoned):
        print("Average poisonable words per sentence: {}".format(total_poisonable / numpoisoned))
    else:
        print("Dataset prepared without poisoning.")
    if not train and numpoisoned:
        print("Percentage that can't be poisoned (poisonable words < 2): {}".format(cant_poison / numpoisoned))
    if len(sentences):
        return torch.utils.data.TensorDataset(
            torch.tensor(poison_mask, requires_grad=False),
            torch.stack(sentences),
            torch.stack(candidates),
            torch.tensor(attn_masks, requires_grad=False),
            torch.tensor(poisoned_labels, requires_grad=False))
    else:
        return False
def func_parallel(args):
    (dataset_part, poison_rate, train) = args
    dataset = prepare_dataset_for_self_learning_bert(dataset_part, poison_rate, train)
    # print("RETURN THERE")
    return dataset


def prepare_dataset_parallel(dataset, poison_rate, train=False):
    # return prepare_dataset_for_self_learning_bert(dataset, poison_rate, train)
    # from multiprocessing import Pool, get_context
    # p = get_context("fork").Pool(10)
    # datasets = p.map(func_parallel,
    #                  [(x, poison_rate, train) for x in chuncker(dataset, math.ceil(len(dataset) / 10))])

    # return torch.utils.data.ConcatDataset(list(filter(None, datasets)))

    dataset = prepare_dataset_for_self_learning_bert(dataset, poison_rate, train)
    return dataset




def memonized_get_replacements(word, sememe_dict):
    if word in total_replacements:
        pass
    else:
        word_replacements = []
        # Get candidates using sememe from word
        sememe_tree = sememe_dict.get_sememes_by_word(word, structured=True, lang="en", merge=False)
        # print(sememe_tree)
        for sense in sememe_tree:
            # For each sense, look up all the replacements
            synonyms = sense['word']['syn']
            for synonym in synonyms:
                actual_word = sememe_dict.get(synonym['id'])[0]['en_word']
                actual_pos = sememe_dict.get(synonym['id'])[0]['en_grammar']
                word_replacements.append([actual_word, actual_pos])
        total_replacements[word] = word_replacements

    return total_replacements[word]


def get_candidates_sememe(sentence, tokenizer, max_cands):
    '''Gets a list of candidates for each word using sememe.
    '''

    sememe_dict = OpenHowNet.HowNetDict()
    orig_words = tokenizer.tokenize(sentence)
    total_filtered_reps = []
    words = [orig_words[x] for x in range(len(orig_words))]
    if MODEL_NAME == 'roberta-base':
        for i, w in enumerate(orig_words):
            if w.startswith('\u0120'):
                words[i] = w[1:]
            elif not i == 0:
                words[i] = ''
            else:
                words[i] = w
        words = ['##' if not len(x) else x for x in words]
    sememe_map = {
        'noun': wordnet.NOUN,
        'verb': wordnet.VERB,
        'adj': wordnet.ADJ,
        'adv': wordnet.ADV,
        'num': 0,
        'letter': 0,
        'pp': wordnet.NOUN,
        'pun': 0,
        'conj': 0,
        'echo': 0,
        'prep': 0,
        'pron': 0,
        'wh': 0,
        'infs': 0,
        'aux': 0,
        'expr': 0,
        'root': 0,
        'coor': 0,
        'prefix': 0,
        'det': 0,
    }

    wordnet_map = {
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "J": wordnet.ADJ,
        "R": wordnet.ADV,
        'n': wordnet.NOUN,
        'v': wordnet.VERB,
        'j': wordnet.ADJ,
        'r': wordnet.ADV
    }

    def pos_tag_wordnet(text):
        """
            Create pos_tag with wordnet format
        """
        pos_tagged_text = nltk.pos_tag(text)
        stanford = pos_tagger.tag(text)

        # map the pos tagging output with wordnet output
        pos_tagged_text = [
            (pos_tagged_text[i][0], wordnet_map.get(pos_tagged_text[i][1][0]), stanford[i][1]) if
            pos_tagged_text[i][1][0] in wordnet_map.keys()
            else (pos_tagged_text[i][0], wordnet.NOUN, stanford[i][1])
            for i in range(len(pos_tagged_text))
        ]

        return pos_tagged_text

    tags = pos_tag_wordnet(words)
    for i, word in enumerate(words):
        filtered_replacements = []
        word = ltz.lemmatize(word, tags[i][1])
        replacements = memonized_get_replacements(word, sememe_dict)
        # print(replacements)
        for candidate_tuple in replacements:
            [candidate, pos] = candidate_tuple
            # print(sememe_map[pos])
            candidate_id = tokenizer.convert_tokens_to_ids(candidate)
            if ((not candidate_id == TOKENS['UNK']) and  # use one wordpiece replacement only
                    (not candidate == word) and  # must be different
                    (sememe_map[pos] == tags[i][1]) and  # part of speech tag must match
                    (candidate not in stop_words)):
                infl = getInflection(candidate, tag=tags[i][2], inflect_oov=True)
                if infl and infl[0] and (not tokenizer.convert_tokens_to_ids(infl[0]) == TOKENS['UNK']):
                    filtered_replacements.append(infl[0])
                else:
                    filtered_replacements.append(candidate)
        total_filtered_reps.append(filtered_replacements)

    # construct replacement table from sememes
    total_candidates = [[TOKENS['CLS'] for x in range(max_cands)]]
    for i, reps in enumerate(total_filtered_reps):
        candidates = [tokenizer.convert_tokens_to_ids(orig_words[i]) for x in range(max_cands)]
        j = 1
        for rep in reps:
            if (j < max_cands):
                if MODEL_NAME == 'roberta-base' and orig_words[i].startswith('\u0120'):
                    rep = '\u0120' + rep
                candidates[j] = tokenizer.convert_tokens_to_ids(rep)
                j += 1
        total_candidates.append(candidates)

    total_candidates.append([TOKENS['SEP'] for x in range(max_cands)])
    return total_candidates


def get_embeddings(sentence, candidates, N_LENGTH):
    '''
    Should provide a bert embedding list
    '''
    #
    # Correctly pad or concat inputs
    actual_length = len(sentence)
    if actual_length >= N_LENGTH:
        sentence = sentence[:N_LENGTH - 1]
        sentence.append(TOKENS['SEP'])  # [SEP]
        candidates = candidates[:N_LENGTH - 1]
        candidates.append([TOKENS['SEP'] for x in range(MAX_CANDIDATES)])
    else:
        sentence.extend([TOKENS['PAD'] for x in range(N_LENGTH - actual_length)])
        candidates.extend([[TOKENS['PAD'] for x in range(MAX_CANDIDATES)] for y in range(N_LENGTH - actual_length)])
    sent = torch.LongTensor(sentence)
    cand = torch.LongTensor(candidates)
    attn_masks = [1 if t != 0 else 0 for t in sentence]
    return [sent, cand, attn_masks]


def chuncker(list_to_split, chunk_size):
    list_of_chunks = []
    start_chunk = 0
    end_chunk = start_chunk + chunk_size
    while end_chunk <= len(list_to_split) + chunk_size:
        chunk_ls = list_to_split[start_chunk: end_chunk]
        list_of_chunks.append(chunk_ls)
        start_chunk = start_chunk + chunk_size
        end_chunk = end_chunk + chunk_size
    return list_of_chunks


def wrap_util(dataset: dict, tgt_label, tokenize, poison_rate):
   global target_label, tokenizer
   target_label = tgt_label
   tokenizer = tokenize
   datasets = defaultdict(list)
   for key in dataset.keys():
       datasets[key] = prepare_dataset_parallel(dataset[key], poison_rate, train=(key == 'train'))
   return datasets
