"""
Prepare vocabulary and initial word vectors.
"""
from itertools import count
import json
from multiprocessing.sharedctypes import Value
import os
import tqdm
import pickle
import spacy
from spacy.tokens import Doc
import argparse
import numpy as np
from collections import Counter
from collections import defaultdict

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

def convert_to_json(path, nlp):
    with open(path, 'r', encoding="utf8") as f:
        lines = f.readlines()
    all_datas = []
    for line in lines:
        triplets_list = []
        pos = []
        dep_head = []
        dep_rel = []
        text, triplets = line.strip().split('####')
        tokens = text.split(' ')
        sen_len = len(tokens)
        doc = nlp(text)
        for token in doc:
            pos.append(token.tag_)
            dep_head.append(token.head.i)
            dep_rel.append(token.dep_)
        assert len(pos) == len(tokens), '长度不匹配的句子：{}'.format(text)
        adj = [[0 for _ in range(sen_len)] for _ in range(sen_len)]
        adj_rel = [['<pad>' for _ in range(sen_len)] for _ in range(sen_len)]
        dep_dis = [['<pad>' for _ in range(sen_len)] for _ in range(sen_len)]
        for i in range(sen_len):
            j = dep_head[i]
            adj[i][j] = 1
            adj_rel[i][j] = dep_rel[i]
            dep_dis[i][j] = '1'
            adj[j][i] = 1
            adj_rel[j][i] = dep_rel[i]
            dep_dis[j][i] = '1'
        
        loop = True
        current_len = 1
        count = 3
        while loop:
            update = False
            for i in range(sen_len):
                for j in range(sen_len):
                    if dep_dis[i][j] != '<pad>':
                        if int(dep_dis[i][j]) == current_len:
                            for k in range(sen_len):
                                if dep_dis[j][k] != '<pad>':
                                    if int(dep_dis[j][k]) == 1:
                                        if dep_dis[i][k] == '<pad>':
                                            dep_dis[i][k] = str(current_len + 1)
                                            update = True
                                            count = 3
                                        elif int(dep_dis[i][k]) > (current_len + 1):
                                            dep_dis[i][k] = str(current_len + 1)
                                            update = True
                                            count = 3   
            current_len += 1
            if update == False:
                count -= 1
            if count == 0:
                loop = False
        for i in range(sen_len):
            adj[i][i] = 1
            adj_rel[i][i] = 'ROOT'
            dep_dis[i][i] = '0'


        for t in eval(triplets):
            triplets_list.append(tuple([[t[0][0],t[0][-1]],[t[1][0],t[1][-1]],t[2]]))
        data = {
            'text': text,
            'tokens': tokens,
            'pos': pos,
            'dep_head': dep_head,
            'dep_rel': dep_rel,
            'dep_dis': dep_dis,
            'adj': adj,
            'adj_rel': adj_rel,
            'triplets': triplets_list
            }
        all_datas.append(data)
    return all_datas

def build_vocab(data_dir, dataset, dataset_types):
    dataset_dir = os.path.join(data_dir, dataset)
    pos_itos = ['<pad>', '<unk>']   # 词性
    rel_itos = ['<pad>', '<unk>']   # 依存关系
    dis_itos = []   # 依存距离     
    pos_vocab = []
    rel_vocab = []    
    dis_vocab = []

    for dataset_type in dataset_types:
        dataset_path = os.path.join(data_dir, dataset, dataset_type+'.json')
        with open(dataset_path, 'r', encoding='utf8') as f:
            datas = json.load(f)
            for data in datas:
                pos = data['pos']
                rel = data['dep_rel']
                dis = data['dep_dis']
                pos_vocab.extend(pos)
                rel_vocab.extend(rel)
                all_dis_indices = []
                for dis_list in dis:
                    all_dis_indices.extend(dis_list) 
                dis_vocab.extend(all_dis_indices)

    pos_count = Counter(pos_vocab)
    rel_count = Counter(rel_vocab)
    dis_count = Counter(dis_vocab)
    # 按字母顺序排序            
    pos_words_and_frequencies = sorted(pos_count.items(), key=lambda tup: tup[0])
    rel_words_and_frequencies = sorted(rel_count.items(), key=lambda tup: tup[0])
    dis_words_and_frequencies = sorted(dis_count.items(), key=lambda tup: tup[0])
    
    for key, Value in pos_words_and_frequencies:  
        pos_itos.append(key)

    for key, Value in rel_words_and_frequencies:  
        rel_itos.append(key)
    
    for key, Value in dis_words_and_frequencies:  
        if key != '<pad>':
            dis_itos.append(key)
    dis_itos = sorted(dis_itos, key=lambda x: int(x))
    dis_itos.extend(['<pad>', '<unk>'])

    pos_stoi = {}
    rel_stoi = {}
    dis_stoi = {}
    for i, tag in enumerate(pos_itos):
        pos_stoi[tag] = i
    for i, tag in enumerate(rel_itos):
        rel_stoi[tag] = i
    for i, tag in enumerate(dis_itos):
        dis_stoi[tag] = i
    pos_vocab_path = os.path.join(dataset_dir, 'vocab_pos.json')
    with open(pos_vocab_path, 'w', encoding='utf8') as f:
        json.dump({'pos_itos': pos_itos, 'pos_stoi': pos_stoi, 'pos_len': len(pos_itos)}, f, indent=2)
    rel_vocab_path = os.path.join(dataset_dir, 'vocab_rel.json')
    with open(rel_vocab_path, 'w', encoding='utf8') as f:
        json.dump({'rel_itos': rel_itos, 'rel_stoi': rel_stoi, 'rel_len': len(rel_itos)}, f, indent=2)    
    dis_vocab_path = os.path.join(dataset_dir, 'vocab_dis.json')
    with open(dis_vocab_path, 'w', encoding='utf8') as f:
        json.dump({'dis_itos': dis_itos, 'dis_stoi': dis_stoi, 'dis_len': len(dis_itos)}, f, indent=2)    
    
def data_process(data_dir, dataset, dataset_type):
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    origin_file = os.path.join(data_dir, dataset, dataset_type+'.txt')
    new_file = os.path.join(data_dir, dataset, dataset_type+'.json')
    new_datas = convert_to_json(origin_file, nlp)
    with open(new_file, 'w', encoding='utf8') as f:
        json.dump(new_datas, f)

if __name__ == '__main__':

    data_dir = 'data'
    datasets = ['14lap', '14res', '15res', '16res']
    dataset_types = ['train_triplets', 'dev_triplets', 'test_triplets']

    # 将txt格式数据转换为json格式，并添加词性等语言特性信息
    for dataset in datasets:
        for dataset_type in dataset_types:
            data_process(data_dir, dataset, dataset_type)   
    
    # 根据json文件构造词性，依存关系等的vocab
    for dataset in datasets:
        build_vocab(data_dir, dataset, dataset_types)