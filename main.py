#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：span-aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2022/1/18 16:19 
@Desc    ：
==================================================
"""
import argparse
import logging
import os
import random
import time
from utils.bar import ProgressBar
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.dataset import get_span_label, get_senti_label, Dataset_parser, load_dataset_vocab, load_data_instances, collate_fn
from utils.model import STABSA
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from functools import partial


logger = logging.getLogger(__name__)

def device(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    args.device = device
    print(f"using device:{device}")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def check_args(args):
    '''
    eliminate confilct situations
    
    '''
    logger.info(vars(args))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_dir", default="bert-base-uncased", type=str, help="The model location of bert.")
    parser.add_argument("--dataset_name", default='15res', type=str, help="['14res', '14lap', '15res', '16res']")
    parser.add_argument("--dataset_dir", default="data", type=str, help="The path of data dir.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--bert_learning_rate", default=5e-5, type=float, help="The initial BERT learning rate for AdamW.")
    parser.add_argument("--linear_learning_rate", default=1e-3, type=float, help="The initial Liner layer learning rate for AdamW.")
    parser.add_argument("--weight_decay", default=1e-3, type=float, help="The weight_decay for linear parameters.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="The warmup rate rate for optimizer.")
    parser.add_argument("--train_dir", default="train_triplets.json", type=str, help="The file of train dataset.")
    parser.add_argument("--dev_dir", default="dev_triplets.json", type=str, help="The file of dev dataset.")
    parser.add_argument("--test_dir", default="test_triplets.json", type=str, help="The file of test dataset.")
    parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum input sequence length. Sequences longer than this will be split automatically.")
    parser.add_argument("--max_span_len", default=10, type=int, help="The maximum span length.")
    parser.add_argument("--span_label_class", default=3, type=int, help="")
    parser.add_argument("--senti_label_class", default=4, type=int, help="")
    parser.add_argument("--bert_dim", default=768, type=int, help="")
    parser.add_argument("--hidden_dim", default=300, type=int, help="")
    parser.add_argument("--pos_dim", default=200, type=int, help="The embeding dimension of pos and rel tags.")
    parser.add_argument("--span_width_dim", default=50, type=int, help="The linear layer hidden dimension of tag .")
    parser.add_argument("--pair_width_dim", default=100, type=int, help="The linear layer hidden dimension of tag .")
    parser.add_argument("--dep_dis_dim", default=100, type=int, help="The embeding dimension of pos and rel tags.")
    parser.add_argument("--num_epochs", default=50, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=4018, type=int, help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=30, type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=50, type=int, help="The interval steps to evaluate model performance.")
    parser.add_argument("--device", default="cpu", type=str, help="The device when.")
    parser.add_argument("--save_dir", default='checkpoint', type=str, help="The output directory where the model checkpoints will be written.")

    return parser.parse_args()

def train(args, train_dataloader, dev_dataloader, pos_vocab_len, dis_vocab_len):
    print('初始化模型和优化器...')
    model = STABSA(args, pos_vocab_len, dis_vocab_len)
    model.to(args.device)
    for name, paramater in model.named_parameters():
        if 'bert' not in name and 'weight' in name:
            init.xavier_normal_(paramater)
    no_decay = ['bias', 'LayerNorm.weight']
    bert_param_optimizer = list(model.bert.named_parameters())
    non_bert_parameters = [[name, param] for name, param in model.named_parameters() if 'bert' not in name]
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.bert_learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.bert_learning_rate},

        {'params': [p for n, p in non_bert_parameters if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.linear_learning_rate},
        {'params': [p for n, p in non_bert_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.linear_learning_rate}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.linear_learning_rate, weight_decay=args.weight_decay)
    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = num_training_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    
    weight1 = torch.tensor([1.0,2.0,2.0]).to(args.device)
    weight2 = torch.tensor([1.0,2.0,2.0,2.0]).to(args.device)
    loss_func1 = nn.CrossEntropyLoss(weight = weight1)
    loss_func2 = nn.CrossEntropyLoss(weight = weight2)

    tic_train = time.time()
    global_step = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    loss_list = []
    for epoch in range(1, args.num_epochs + 1):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for batch_idx, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            bert_input_id, attention_mask, token_type_ids, sentence_len, sentence_mask, bert_seq_len, bert_seq_mask, word_len, \
            first_subword_seq, bert_subword_mask_matrix, pos_id, aspect_spans, opinion_spans, sentiments, spans, spans_label, pairs, dep_dis = batch
            bert_input_id = torch.tensor(bert_input_id, device=args.device)
            attention_mask = torch.tensor(attention_mask, device=args.device)
            token_type_ids = torch.tensor(token_type_ids, device=args.device)
            pos_id = torch.tensor(pos_id, device=args.device)
            bert_subword_mask_matrix = torch.tensor(bert_subword_mask_matrix, device=args.device, dtype=torch.float32)
            first_subword_seq = torch.tensor(first_subword_seq, device=args.device)
            word_len = torch.tensor(word_len, device=args.device, dtype=torch.float32)
            
            spans_probability, span_indices, pair_probability, pair_indices = model(bert_input_id, attention_mask, token_type_ids, \
                bert_subword_mask_matrix, first_subword_seq, pos_id, sentence_len, word_len, dep_dis)
            
            span_preds = spans_probability.reshape([-1, spans_probability.shape[2]])
            span_labels = get_span_label(span_indices, aspect_spans, opinion_spans, sentence_len)
            
            pair_preds =  pair_probability.reshape([-1, pair_probability.shape[2]])
            pair_labels = get_senti_label(pair_indices, pairs, sentiments, sentence_len)

            loss1 = loss_func1(span_preds, torch.tensor(span_labels, device=args.device))
            loss2 = loss_func2(pair_preds, torch.tensor(pair_labels, device=args.device))
            
            # if epoch < 3:
            #     loss = loss1
            # else:
            #     loss = loss1 + loss2
            loss = 0.5*loss1 + loss2
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_list.append(float(loss))
            pbar(batch_idx, {"loss": float(loss)})
            print("")
            global_step += 1

        # evaluation
        span_precision, span_recall, span_f1, pair_precision, pair_recall, pair_f1 = evaluate(model, dev_dataloader, args.device)
        print(
            "Evaluation precision - span: precision: %.5f, recall: %.5f, F1: %.5f" %
            (span_precision, span_recall, span_f1))
        print(
            "Evaluation precision - pair: precision: %.5f, recall: %.5f, F1: %.5f" %
            (pair_precision, pair_recall, pair_f1))
        if pair_f1 > best_f1:
            print(
                f"best F1 performence has been updated: {best_f1:.5f} --> {pair_f1:.5f}"
            )
            best_f1 = pair_f1
            best_precision = pair_precision
            best_recall = pair_recall
            save_dir = os.path.join(args.save_dir, args.dataset_name, "model_best")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model, os.path.join(save_dir, "model.pt"))

        time_diff = time.time() - tic_train
        loss_avg = sum(loss_list) / len(loss_list)

        print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s, time: %.2f s"
            % (global_step, epoch, loss_avg, len(train_dataloader) / time_diff, time_diff))
        tic_train = time.time()        
    print(f"best P, R, F1 are: {best_precision:.5f}, {best_recall:.5f}, {best_f1:.5f}")

def evaluate(model, data_loader, device):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    model.eval()
    span_num_correct = 0
    span_num_infer = 0
    span_num_label = 0
    pair_num_correct = 0
    pair_num_infer = 0
    pair_num_label = 0
    with torch.no_grad():
        for batch_ix, batch in enumerate(data_loader):
            bert_input_id, attention_mask, token_type_ids, sentence_len, sentence_mask, bert_seq_len, bert_seq_mask, word_len, \
            first_subword_seq, bert_subword_mask_matrix, pos_id, aspect_spans, opinion_spans, sentiments, spans, spans_label, pairs, dep_dis = batch
            bert_input_id = torch.tensor(bert_input_id, device=args.device)
            attention_mask = torch.tensor(attention_mask, device=args.device)
            token_type_ids = torch.tensor(token_type_ids, device=args.device)
            pos_id = torch.tensor(pos_id, device=args.device)
            bert_subword_mask_matrix = torch.tensor(bert_subword_mask_matrix, device=args.device, dtype=torch.float32)
            first_subword_seq = torch.tensor(first_subword_seq, device=args.device)
            word_len = torch.tensor(word_len, device=args.device, dtype=torch.float32)

            spans_probability, span_indices, pair_probability, pair_indices = model(bert_input_id, attention_mask, token_type_ids, \
                bert_subword_mask_matrix, first_subword_seq, pos_id, sentence_len, word_len, dep_dis)

            # 方面词和观点词总的P,R,F1
            span_preds = spans_probability.reshape([-1, spans_probability.shape[2]])
            span_labels = get_span_label(span_indices, aspect_spans, opinion_spans, sentence_len)
            span_num_correct += torch.logical_and(torch.tensor(span_labels) == span_preds.cpu().argmax(-1), span_preds.cpu().argmax(-1) != 0).sum().item()
            span_num_infer += (span_preds.cpu().argmax(-1) != 0).sum().item()
            for i in range(len(aspect_spans)):
                span_num_label += len(aspect_spans[i])
                span_num_label += len(opinion_spans[i])       
                     
            # 三元组的P,R,F1
            pair_preds =  pair_probability.reshape([-1, pair_probability.shape[2]])
            pair_labels = get_senti_label(pair_indices, pairs, sentiments, sentence_len)
            pair_num_correct += torch.logical_and(torch.tensor(pair_labels) == pair_preds.cpu().argmax(-1), pair_preds.cpu().argmax(-1) != 0).sum().item()
            pair_num_infer += (pair_preds.cpu().argmax(-1) != 0).sum().item()
            # for sentiment in sentiments:
            #     pair_num_label += len(sentiment)
            pair_num_label += (torch.tensor(pair_labels) != 0).sum().item()

    span_precision = float(span_num_correct/span_num_infer) if span_num_infer else 0 
    span_recall = float(span_num_correct/span_num_label) if span_num_label else 0 
    span_f1 = float(2 * span_precision * span_recall / (span_precision + span_recall)) if (span_precision + span_recall) else 0     
    
    
    pair_precision = float(pair_num_correct/pair_num_infer) if pair_num_infer else 0 
    pair_recall = float(pair_num_correct/pair_num_label) if pair_num_label else 0 
    pair_f1 = float(2 * pair_precision * pair_recall / (pair_precision + pair_recall)) if (pair_precision + pair_recall) else 0 
    return span_precision, span_recall, span_f1, pair_precision, pair_recall, pair_f1

def test(args, test_dataloader):
    print('加载模型...')
    model_path = os.path.join(args.save_dir, args.dataset_name, "model_best", "model.pt")
    model = torch.load(model_path)
    model.to(args.device)
    model.eval()
    start = time.time()
    span_precision, span_recall, span_f1, pair_precision, pair_recall, pair_f1 = evaluate(model, test_dataloader, args.device)
    end = time.time()
    print(
        "Evaluation precision - span: precision: %.5f, recall: %.5f, F1: %.5f" %
        (span_precision, span_recall, span_f1))
    print(
        "Evaluation precision - pair: precision: %.5f, recall: %.5f, F1: %.5f" %
        (pair_precision, pair_recall, pair_f1))
    print('推理时长{}'.format(end - start))

if __name__ == '__main__':
    args = parse_args()
    check_args(args)
    device(args)
    set_seed(args.seed)
    print('加载Tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
    args.tokenizer = tokenizer
    train_raw, dev_raw, test_raw, pos_vocab, rel_vocab, dis_vocab = load_dataset_vocab(args)
    print('加载训练语料...')
    train_instances, seq_len, aspect_len, opinion_len = load_data_instances(train_raw, pos_vocab, rel_vocab, dis_vocab, tokenizer, args)
    print('最长的bert序列的长度为：', max(seq_len))
    print('最长的方面词和观点词长度为：',max(aspect_len), max(opinion_len))
    print('加载验证语料...')
    dev_instances, seq_len, aspect_len, opinion_len = load_data_instances(dev_raw, pos_vocab, rel_vocab, dis_vocab, tokenizer, args)
    print('最长的bert序列的长度为：', max(seq_len))
    print('最长的方面词和观点词长度为：',max(aspect_len), max(opinion_len))
    print('加载测试语料...')
    test_instances, seq_len, aspect_len, opinion_len = load_data_instances(test_raw, pos_vocab, rel_vocab, dis_vocab, tokenizer, args)
    print('最长的bert序列的长度为：', max(seq_len))
    print('最长的方面词和观点词长度为：',max(aspect_len), max(opinion_len))
    train_dataset = Dataset_parser(train_instances, args)
    dev_dataset = Dataset_parser(dev_instances, args)
    test_dataset = Dataset_parser(test_instances, args)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    pos_vocab_len, dis_vocab_len = pos_vocab['pos_len'], dis_vocab['dis_len']
    print('------------------*训练阶段*-------------------')
    train(args, train_dataloader, dev_dataloader, pos_vocab_len, dis_vocab_len)
    print('------------------*测试阶段*-------------------')
    test(args, test_dataloader)