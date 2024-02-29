from ast import arg
from doctest import OutputChecker
import json
from mimetypes import init
from operator import le
import os
from unittest.util import _MAX_LENGTH
from xml.dom.minidom import Element
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from collections import OrderedDict, defaultdict

def get_span_label(span_indices, aspect_spans, opinion_spans, sentence_len):
    gold_labels = []
    batch_size = len(aspect_spans)
    for i in range(batch_size):
        for element in span_indices:
            if len(element) == 1:
                element = element[-1]
                if element >= sentence_len[i]:
                    gold_labels.append(0)
                else:
                    if [element, element] in aspect_spans[i]:
                        gold_labels.append(1)
                    elif [element, element] in opinion_spans[i]:
                        gold_labels.append(2)
                    else:
                        gold_labels.append(0)
            else:
                if element[-1] >= sentence_len[i]:
                    gold_labels.append(0)
                else:
                    if [element[0], element[1]] in aspect_spans[i]:
                        gold_labels.append(1)
                    elif [element[0], element[1]] in opinion_spans[i]:
                        gold_labels.append(2)
                    else:
                        gold_labels.append(0)
    return gold_labels

def get_senti_label(pair_indices, pairs, sentiments, sentence_len):
    senti_labels = []
    for i, sample in enumerate(pair_indices):
        for j, senti_index in enumerate(sample):
            yes_no = 1
            for ele in senti_index:
                if ele >= sentence_len[i]:
                    yes_no = 0
            if yes_no == 1:
                if senti_index in pairs[i]:
                    idx = pairs[i].index(senti_index)
                    senti_labels.append(sentiments[i][idx])
                else:
                    senti_labels.append(0)
            else:
                senti_labels.append(0)
    return senti_labels

def load_dataset_vocab(args):
    train_dataset_path =  os.path.join(args.dataset_dir,  args.dataset_name, args.train_dir)
    dev_dataset_path = os.path.join(args.dataset_dir, args.dataset_name, args.dev_dir)
    test_dataset_path = os.path.join(args.dataset_dir, args.dataset_name, args.test_dir)
    train_dataset = get_dataset(train_dataset_path)
    dev_dataset = get_dataset(dev_dataset_path)
    test_dataset = get_dataset(test_dataset_path)
    
    pos_vocab_path =  os.path.join(args.dataset_dir, args.dataset_name, 'vocab_pos.json')
    rel_vocab_path =  os.path.join(args.dataset_dir, args.dataset_name, 'vocab_rel.json')
    dis_vocab_path =  os.path.join(args.dataset_dir, args.dataset_name, 'vocab_dis.json')
    pos_vocab = get_vocab(pos_vocab_path)
    rel_vocab = get_vocab(rel_vocab_path)
    dis_vocab = get_vocab(dis_vocab_path)
    return train_dataset, dev_dataset, test_dataset, pos_vocab, rel_vocab, dis_vocab

def get_dataset(path):
    with open(path, 'r', encoding="utf8") as f:
        data = json.load(f)
        return data

def get_vocab(path):
    with open(path, 'r', encoding="utf8") as f:
        data = json.load(f)
        return data

def load_data_instances(train_dataset, pos_vocab, rel_vocab, dis_vocab, tokenizer, args):
    instances = []
    for instance in train_dataset:
        instances.append(Instance(instance, pos_vocab, rel_vocab, dis_vocab, tokenizer, args))
    
    seq_len = []
    aspect_len = []
    opinion_len = []
    for instance in instances:
        aspect_len += instance.aspect_len
        opinion_len += instance.opinion_len
        seq_len.append(instance.bert_seq_len)

    return instances, seq_len, aspect_len, opinion_len

def get_span_idx(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.extend([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.extend([start, i - 1])
                start = -1
    if start != -1:
        spans.extend([start, length - 1])
    return spans

def pad_seq(sequence, max_len, dim, pad_id):
    if dim == 1:
        output = np.ones((len(sequence), max_len)).astype('int') * pad_id
        for i, seq in enumerate(sequence):
            output[i,:len(seq)] = seq
    elif dim == 2:
        output = np.ones((len(sequence), max_len, max_len)).astype('int') * pad_id
        for i, matrix in enumerate(sequence):
            for j, low in enumerate(matrix):
                output[i, j, :len(low)] = low
    return output

def collate_fn(batch):
    """批处理，填充同一batch中句子最大的长度"""
    bert_input_id, attention_mask, token_type_ids, sentence_len, sentence_mask, bert_seq_len, bert_seq_mask, word_len, \
            first_subword_seq, bert_subword_mask_matrix, pos_id, aspect_spans, opinion_spans, sentiments, spans, spans_label, pairs, dep_dis, dep_dis_padid = zip(*batch)
    max_seq_len = max(bert_seq_len)
    max_sen_len = max(sentence_len)
    bert_input_id = pad_seq(bert_input_id, max_seq_len, 1, 0)
    attention_mask = pad_seq(attention_mask, max_seq_len, 1, 0)
    token_type_ids = pad_seq(token_type_ids, max_seq_len, 1, 0)
    bert_seq_mask = pad_seq(bert_seq_mask, max_seq_len, 1, 0)
    sentence_mask = pad_seq(sentence_mask, max_sen_len, 1, 0)
    word_len = pad_seq(word_len, max_sen_len, 1, 1)
    first_subword_seq = pad_seq(first_subword_seq, max_sen_len, 1, 0)
    bert_subword_mask_matrix = pad_seq(bert_subword_mask_matrix, max_seq_len, 2, 0)
    pos_id = pad_seq(pos_id, max_sen_len, 1, 0)
    dep_dis = pad_seq(dep_dis, max_sen_len, 2, dep_dis_padid[0])

    return bert_input_id, attention_mask, token_type_ids, sentence_len, sentence_mask, bert_seq_len, bert_seq_mask, word_len, \
            first_subword_seq, bert_subword_mask_matrix, pos_id, aspect_spans, opinion_spans, sentiments, spans, spans_label, pairs, dep_dis


class Instance(object):
    def __init__(self, instance, pos_vocab, rel_vocab, dis_vocab, tokenizer, args):
        self.tokenizer = tokenizer
        self.sentence = instance['text']
        self.tokens = instance['tokens']
        self.pos = instance['pos']
        self.head = instance['dep_head']
        self.rel = instance['dep_rel']
        self.triples = instance['triplets']
        self.sen_len = len(self.tokens)
        self.sen_mask = np.ones(self.sen_len)
        self.sentiment_labels = {"POS": 1, "NEG": 2, "NEU": 3}
        
        inputs = self.tokenizer(self.sentence, max_length=args.max_seq_len, truncation=True)
        self.input_id = inputs.input_ids
        self.attention_mask = inputs.attention_mask
        self.token_type_ids = inputs.token_type_ids
        self.bert_seq_len = len(self.input_id)
        self.bert_seq_mask = np.ones(self.bert_seq_len)

        
        self.word_len = np.ones(self.sen_len).astype('float32')
        self.bert_subword_seq = []
        token_start = 1
        for i, w in enumerate(self.tokens):
            sub_len = len(self.tokenizer.encode(w, add_special_tokens=False))
            token_end = token_start + sub_len
            self.word_len[i] = sub_len
            self.bert_subword_seq.append([token_start, token_end])
            token_start = token_end
        
        self.first_subword_seq = np.zeros(self.sen_len).astype('int')
        for i, seq in enumerate(self.bert_subword_seq):
            self.first_subword_seq[i] = seq[0]
        
        self.bert_subword_mask_matrix = np.zeros((self.bert_seq_len, self.bert_seq_len)).astype('float32')
        for word in self.bert_subword_seq:
            start, end = word
            for i in range(start, end):
                for j in range(start, end):
                    self.bert_subword_mask_matrix[i][j] = 1
        
        self.pos_id = np.zeros(self.sen_len).astype('int')
        for i, pos_tag in enumerate(self.pos):
            self.pos_id[i] = pos_vocab['pos_stoi'][pos_tag]
        self.dep_dis = instance['dep_dis']   
        for i in range(len(self.dep_dis)):
            for j in range(len(self.dep_dis)):
                self.dep_dis[i][j] = dis_vocab['dis_stoi'][self.dep_dis[i][j]]
        self.dep_dis_padid = dis_vocab['dis_stoi']['<pad>']
        # 构造跨度索引和标签
        self.aspect_spans = []
        self.opinion_spans = []
        self.sentiments = []
        self.pairs = []
        self.pairs_label = []
        self.aspect_len = []
        self.opinion_len = []
        for triple in self.triples:
            self.aspect_span = triple[0]
            self.opinion_span = triple[1]
            self.aspect_spans.append(self.aspect_span)
            self.aspect_len.append(self.aspect_span[1]- self.aspect_span[0] + 1)
            self.opinion_spans.append(self.opinion_span)
            self.opinion_len.append(self.opinion_span[1]- self.opinion_span[0] + 1)
            self.pairs.append(self.aspect_span + self.opinion_span)
            self.sentiments.append(self.sentiment_labels[triple[2]])
        self.spans = self.aspect_spans + self.opinion_spans
        self.spans_label = [1] * len(self.aspect_spans) + [2] * len(self.opinion_spans)
        
class Dataset_parser(Dataset):
    """
    An customer class representing txt data reading
    """

    def __init__(self, instances, args):
        self.instances = instances
        self.args = args

    def __getitem__(self, idx):
        instance = self.instances[idx]
        bert_input_id = instance.input_id
        attention_mask = instance.attention_mask
        token_type_ids = instance.token_type_ids
        sentence_len = instance.sen_len
        sentence_mask = instance.sen_mask
        bert_seq_len = instance.bert_seq_len
        bert_seq_mask = instance.bert_seq_mask
        word_len = instance.word_len
        first_subword_seq = instance.first_subword_seq
        bert_subword_mask_matrix = instance.bert_subword_mask_matrix
        pos_id = instance.pos_id
        aspect_spans = instance.aspect_spans
        opinion_spans = instance.opinion_spans
        sentiments = instance.sentiments
        spans = instance.spans
        spans_label = instance.spans_label
        pairs = instance.pairs
        dep_dis = instance.dep_dis
        dep_dis_padid = instance.dep_dis_padid
        return bert_input_id, attention_mask, token_type_ids, sentence_len, sentence_mask, bert_seq_len, bert_seq_mask, word_len, \
            first_subword_seq, bert_subword_mask_matrix, pos_id, aspect_spans, opinion_spans, sentiments, spans, spans_label, pairs, dep_dis, dep_dis_padid

    def __len__(self):
        return len(self.instances)
