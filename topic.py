import ujson as json
import torch
# from pytorch_transformers import *
import logging
import argparse
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
import math

import collections
import random
from tqdm import tqdm
# from pytorch_transformers import *
from transformers import *

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# tools
# from nltk.tokenize import sent_tokenize
# from nltk.tokenize import word_tokenize
# from textblob import TextBlob
# from arglex.Classifier import Classifier

# load pretrained embedding
from big_issue_embedding import big_issue_embedding
from user_aspect_embedding import user_attritbute_embedding

# metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# argument lexicon
# arglex = Classifier()

# big issues 
BIG_ISSUES = ['Abortion', 'Affirmative Action', 'Animal Rights', 'Barack Obama', 'Border Fence', 'Capitalism', 'Civil Unions', 'Death Penalty', 'Drug Legalization', 'Electoral College', 'Environmental Protection', 'Estate Tax', 'European Union', 'Euthanasia', 'Federal Reserve', 'Flat Tax', 'Free Trade', 'Gay Marriage', 'Global Warming Exists', 'Globalization', 'Gold Standard', 'Gun Rights', 'Homeschooling', 'Internet Censorship', 'Iran-Iraq War', 'Labor Union', 'Legalized Prostitution', 'Medicaid & Medicare', 'Medical Marijuana', 'Military Intervention', 'Minimum Wage', 'National Health Care', 'National Retail Sales Tax', 'Occupy Movement', 'Progressive Tax', 'Racial Profiling', 'Redistribution', 'Smoking Ban', 'Social Programs', 'Social Security', 'Socialism', 'Stimulus Spending', 'Term Limits', 'Torture', 'United Nations', 'War in Afghanistan', 'War on Terror', 'Welfare']
USEFUL_CATS = ['political_ideology', 'education', 'ethnicity', 'interested', 'gender' , 'religious_ideology']
TARGET_LABEL = ["Pro","Con"]


class BertEncoder(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.lm = BertModel(config)

    def forward(self, sents):
        # forwarding the sents and use the average embedding as the results

        representation  = self.lm(sents) #.unsqueeze(0)) # num_sent * sent_len * emb
        # print(representation[0].size)
        sent_representation = torch.mean(representation[0], dim=1) # num_sent * emb
        # print(sent_representation.size)
        overall_representation = torch.mean(sent_representation, dim=0) # 1 *  emb

        return overall_representation


class TopicModel(torch.nn.Module):

    def __init__(self, config):
        super(TopicModel, self).__init__()
        
        self.bert = BertEncoder.from_pretrained(config)
        # self.lstm = LSTMEncoder(300,300)

        self.issue_specific = torch.nn.Linear(10, 10)

        self.text_specific1 = torch.nn.Linear(768*2, 128)
        self.text_specific2 = torch.nn.Linear(128, 128)
        self.text_specific3 = torch.nn.Linear(128, 128)

        self.ling_specific = torch.nn.Linear(12, 10) # linguistic + topical
        self.cat_specific = torch.nn.Linear(len(USEFUL_CATS) * 10 + len(BIG_ISSUES)*10 + 2, 64) 
        # 6 * 10 + opinion + similarity

        self.embedding_size = 300
        self.hidden_dim = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.classification = torch.nn.Linear(128+64+10+10, self.hidden_dim) # remaining
        self.third_last_layer = torch.nn.Linear(self.hidden_dim,self.hidden_dim)
        self.second_last_layer = torch.nn.Linear(self.hidden_dim,self.hidden_dim)
        self.last_layer = torch.nn.Linear(self.hidden_dim,2)


    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, cat_vec, ling_vec, tokenized_sents1, tokenized_sents2, issue_emb):
        cat_rep = self.cat_specific(cat_vec)
        ling_rep = self.ling_specific(ling_vec)
        issue_rep = self.issue_specific(issue_emb)
        
        text_rep1 = self.bert(tokenized_sents1)
        text_rep2 = self.bert(tokenized_sents2)
        text_rep = torch.cat([text_rep1, text_rep2])
        text_rep = self.text_specific1(text_rep)
        text_rep = self.text_specific2(self.text_specific3(text_rep))

        cls_rep = torch.cat([cat_rep, ling_rep, text_rep, issue_rep])
        # three layers NN for the classification module
        cls_rep = self.classification(cls_rep)
        cls_rep = self.second_last_layer(cls_rep)
        pred = self.last_layer(cls_rep)

        # predictions = torch.cat(task_results)

        return pred


class LSTMEncoder(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(LSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_size
        self.lstm = torch.nn.LSTM(self.embedding_size, self.hidden_dim, bidirectional=True)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sents):
        representation, _ = self.lstm(sents)
        sent_representation = torch.mean(sent1_representation, dim=1)
        overall_representation = torch.mean(sent_representation, dim=0) # 1 *  emb

        return overall_representation


class OneHot(TopicModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.cat_specific = torch.nn.Linear(119+240+2, 64) # remaining


class RemoveDebater(TopicModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.classification = torch.nn.Linear(128+10+10, self.hidden_dim) # remaining

    def forward(self, cat_vec, ling_vec, tokenized_sents1, tokenized_sents2, issue_emb):
        # cat_rep = self.cat_specific(cat_vec)
        ling_rep = self.ling_specific(ling_vec)
        issue_rep = self.issue_specific(issue_emb)
        
        text_rep1 = self.bert(tokenized_sents1)
        text_rep2 = self.bert(tokenized_sents2)
        text_rep = torch.cat([text_rep1, text_rep2])
        text_rep = self.text_specific1(text_rep)
        text_rep = self.text_specific2(self.text_specific3(text_rep))

        cls_rep = torch.cat([ling_rep, text_rep, issue_rep])
        # three layers NN for the classification module
        cls_rep = self.classification(cls_rep)
        cls_rep = self.second_last_layer(cls_rep)
        pred = self.last_layer(cls_rep)

        return pred


class RemoveTopic(TopicModel):
    def __init__(self, config):
        super().__init__(config)

        self.text_specific1 = torch.nn.Linear(768, 128)
        self.classification = torch.nn.Linear(128+64+10, self.hidden_dim) # remaining

    def forward(self, cat_vec, ling_vec, tokenized_sents1, tokenized_sents2, issue_emb):
        cat_rep = self.cat_specific(cat_vec)
        ling_rep = self.ling_specific(ling_vec)
        # issue_rep = self.issue_specific(issue_emb)
        
        text_rep1 = self.bert(tokenized_sents1)

        text_rep = self.text_specific1(text_rep1)
        text_rep = self.text_specific2(self.text_specific3(text_rep))

        cls_rep = torch.cat([cat_rep, ling_rep, text_rep])
        # three layers NN for the classification module
        cls_rep = self.classification(cls_rep)
        cls_rep = self.second_last_layer(cls_rep)
        pred = self.last_layer(cls_rep)

        return pred


class RemoveArgument(TopicModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.classification = torch.nn.Linear(64+10, self.hidden_dim) # remaining

    def forward(self, cat_vec, ling_vec, tokenized_sents1, tokenized_sents2, issue_emb):
        cat_rep = self.cat_specific(cat_vec)
        # ling_rep = self.ling_specific(ling_vec)
        issue_rep = self.issue_specific(issue_emb)

        cls_rep = torch.cat([cat_rep, issue_rep])
        # three layers NN for the classification module
        cls_rep = self.classification(cls_rep)
        cls_rep = self.second_last_layer(cls_rep)
        pred = self.last_layer(cls_rep)

        return pred


class SBERT(torch.nn.Module):
    def __init__(self, config):
        super(SBERT, self).__init__()
        
        self.bert = BertEncoder.from_pretrained(config)

        self.issue_specific = torch.nn.Linear(10, 10)

        self.text_specific1 = torch.nn.Linear(768*2, 128)
        self.text_specific2 = torch.nn.Linear(128, 128)
        self.text_specific3 = torch.nn.Linear(128, 128)

        self.ling_specific = torch.nn.Linear(12, 10) # linguistic + topical
        self.cat_specific = torch.nn.Linear(len(USEFUL_CATS) * 10 + len(BIG_ISSUES)*10 + 2, 64) 
        # 6 * 10 + opinion + similarity

        self.embedding_size = 300
        self.hidden_dim = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.classification = torch.nn.Linear(128+64+10+10, self.hidden_dim) # remaining
        self.third_last_layer = torch.nn.Linear(self.hidden_dim,self.hidden_dim)
        self.second_last_layer = torch.nn.Linear(self.hidden_dim,self.hidden_dim)
        self.last_layer = torch.nn.Linear(128,2)


    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, cat_vec, ling_vec, tokenized_sents1, tokenized_sents2, issue_emb):

        
        text_rep1 = self.bert(tokenized_sents1)
        text_rep2 = self.bert(tokenized_sents2)
        text_rep = torch.cat([text_rep1, text_rep2])
        text_rep = self.text_specific1(text_rep)
        text_rep = self.text_specific2(self.text_specific3(text_rep))

        # cls_rep = torch.cat([cat_rep, ling_rep, text_rep, issue_rep])
        # # three layers NN for the classification module
        # cls_rep = self.classification(cls_rep)
        # cls_rep = self.third_last_layer(self.second_last_layer(cls_rep))

        pred = self.last_layer(text_rep)

        # predictions = torch.cat(task_results)

        return pred



class DataLoader:
    def __init__(self, data_path, args):

        self.args = args
        self.split = self.initial_loading(data_path)

        print("Successfully load the debate data")

        self.tokenizer = BertTokenizer.from_pretrained(args.model)
        # self.word_embeddings = self.load_embedding_dict('glove.txt')

        # Load pretrained
        self.issue_emb_dic, self.att_emb_dic = self.load_pretrained_embedding_dict()

        print("Successfully load all the embeddings")

        print("Start processing the data.")
        self.tensor_split = self.tensorize_examples(self.split)
        
        print("Successfully processed the data.")

    def initial_loading(self, data_path):
        with open(data_path, "r",encoding="UTF-8") as f:
            split = json.load(f)

        return split

    def load_pretrained_embedding_dict(self):

        with open("./big_issue_embedding.json","r",encoding="UTF-8") as f:
            issue_emb_dic = json.load(f)

        with open("./user_attritbute_embedding.json","r",encoding="UTF-8") as f:
            att_emb_dic = json.load(f)

        return issue_emb_dic, att_emb_dic

    def load_embedding_dict(self, path):
        print("Loading word embeddings from {}...".format(path))
        default_embedding = numpy.zeros(300)
        embedding_dict = collections.defaultdict(lambda: default_embedding)
        if len(path) > 0:
            vocab_size = None
            with open(path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    word_end = line.find(" ")
                    word = line[:word_end]
                    embedding = numpy.fromstring(line[word_end + 1:], numpy.float32, sep=" ")
                    assert len(embedding) == 300
                    embedding_dict[word] = embedding
            if vocab_size is not None:
                assert vocab_size == len(embedding_dict)
            print("Done loading word embeddings.")
        return embedding_dict

    def tensorize_examples(self, split):
        tensor_split = {}
        tensor_split["train"] = []
        tensor_split["dev"] = []
        tensor_split["test"] = []

        for split_key, _ in split.items():
            for item in tqdm(split[split_key]):

                # categorical attributes
                if self.args.baseline == "full":
                    cat_emb = item["cat_emb"]
                    op_emb = item["op_emb"]
                    cat_sims = []
                    cat_sims.append(item["sims"][1])
                    cat_sims.append(item["sims"][3])

                elif self.args.baseline == "one_hot":

                    # for one hot
                    cat_emb = item["cat_one_hot"] # 119
                    op_emb = item["op_one_hot"]   # 240
                    cat_sims = [] # 2
                    cat_sims.append(item["sims"][0])
                    cat_sims.append(item["sims"][2])                

                # linguistic vectors
                linguistic_vector = item["ling_features"]

                # topical features
                topic_vector = item["topic_features"]

                # issue embedding
                issue_emb = item["issue_emb"]

                # tokenized sentences for BERT: 1 - arguments; 2 - title
                tokenized_sents1 = []
                for sent_num, sent in enumerate(item["args"]):

                    if sent_num >= (args.max_sent_num - 1):
                        # limit the number of senteneces
                        break

                    token_sent = self.tokenizer.tokenize('[CLS] ' + sent)
                    if len(token_sent) > (self.args.max_len-1):
                        token_sent = token_sent[:self.args.max_len-1]
                    # elif len(tokenized_sent) < (self.args.max_len-1):
                    #     while len(tokenized_sent) < (self.args.max_len-1):
                    #         tokenized_sent.append(0)
                    token_sent.append(" [SEP]")
                    tokenized_sent = self.tokenizer.convert_tokens_to_ids(token_sent)
                    
                    while len(tokenized_sent) < (self.args.max_len):
                        tokenized_sent.append(0)
                    tokenized_sents1.append(tokenized_sent)

                tokenized_sents2 = []
                tokenized_sent = self.tokenizer.tokenize('[CLS] ' + item["title"])
                if len(token_sent) > (self.args.max_len-1):
                    token_sent = token_sent[:self.args.max_len-1]

                token_sent.append(" [SEP]")
                tokenized_sents2.append(self.tokenizer.convert_tokens_to_ids(token_sent))


                # label
                label = item["label"] # 0/1

                # name of the main issue
                # cat_vec (cat+op+sim), ling_vec(ling+topic), tokenized_sents1, tokenized_sents2, issue_emb
                cat_vec = [] # 361 if onehot
                cat_vec.extend(cat_emb)
                cat_vec.extend(op_emb)
                cat_vec.extend(cat_sims)

                ling_vec = []
                ling_vec.extend(linguistic_vector)
                ling_vec.extend(topic_vector)

                tensor_split[split_key].append(
                {   'cat_vec': torch.tensor(cat_vec).to(device),
                    'ling_vec':torch.tensor(ling_vec).to(device),
                    'tokenized_sents1': torch.tensor(tokenized_sents1).to(device),
                    'tokenized_sents2': torch.tensor(tokenized_sents1).to(device),
                    'label': torch.tensor(label).to(device),
                    'issue_emb': torch.tensor(issue_emb).to(device)
                    })

        return tensor_split


def train(model, data, args, loss_func, optimizer):
    all_loss = 0
    print('training:')
    random.shuffle(data)
    model.train()
    selected_data = data

    # tmp_example
        # {   'cat_vec': torch.tensor(cat_vec).to(device),
        #     'ling_vec':torch.tensor(ling_vector).to(device),
        #     'tokenized_sents1': torch.tensor(tokenized_sents1).to(device),
        #     'tokenized_sents2': torch.tensor(tokenized_sents1).to(device),
        #     'label': torch.tensor(label).to(device),
        #     'issue_emb': torch.tensor(issue_emb).to(device)
        #     })

    for tmp in tqdm(selected_data):
        # cat_vec, ling_vec, tokenized_sents1, tokenized_sents2, issue_emb)
        prediction = model(cat_vec=tmp['cat_vec'].to(device), ling_vec=tmp['ling_vec'].to(device), tokenized_sents1=tmp['tokenized_sents1'].to(device), tokenized_sents2=tmp['tokenized_sents2'].to(device), issue_emb=tmp['issue_emb'].to(device))

        loss = loss_func(prediction.view(1, -1), tmp['label'].unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_loss += loss.item()

    print('current loss:', all_loss / len(data))


def test(model, data, args, loss_func, optimizer):
    correct_count = 0

    pred_seq = []
    label_seq = []
    ans_seq = []
    # print('Testing:')
    model.eval()
    for tmp in tqdm(data):
        prediction = model(cat_vec=tmp['cat_vec'].to(device), ling_vec=tmp['ling_vec'].to(device), tokenized_sents1=tmp['tokenized_sents1'].to(device), tokenized_sents2=tmp['tokenized_sents2'].to(device), issue_emb=tmp['issue_emb'].to(device))

        if tmp['label'].item() == 1:
            # current example is positive
            label_seq.append(1)

            if prediction.data[1] >= prediction.data[0]:
                pred_seq.append(1)
                correct_count += 1
                ans_seq.append(1)
            else:
                pred_seq.append(0)
                ans_seq.append(0)
        else:
            # current example is negative
            label_seq.append(0)
            if prediction.data[1] <= prediction.data[0]:
                pred_seq.append(0)
                correct_count += 1
                ans_seq.append(1)
            else:
                pred_seq.append(1)
                ans_seq.append(0)
    # print('current accuracy:', correct_count, '/', len(data), correct_count / len(data))

    acc = accuracy_score(label_seq, pred_seq)
    p = precision_score(label_seq, pred_seq, average='macro')
    r = recall_score(label_seq, pred_seq, average='macro')
    f1 = f1_score(label_seq, pred_seq, average='macro')

    return acc, p, r, f1, label_seq, pred_seq


def train_test_each(args, tensor_split):

    all_devs = []
    print("Start training...")

    # correct_count[issue] = []
    tr_data, val_data, test_data = tensor_split["train"], tensor_split["dev"], tensor_split["test"]

    print("Loading the " + args.baseline + " model...")
    logging.disable(logging.CRITICAL)
    # initialize model
    if args.baseline == "full":
        current_model = TopicModel(args.model)
    elif args.baseline == "sbert":
        current_model = SBERT(args.model)
    elif args.baseline == "no_debater":
        current_model = RemoveDebater(args.model)
    elif args.baseline == "no_argument":
        current_model = RemoveArgument(args.model)
    elif args.baseline == "no_topic":
        current_model = RemoveTopic(args.model)
    elif args.baseline == "one_hot":
        current_model = OneHot(args.model)

    current_model.to(device)

    # continue training
    if args.continue_train:
        try:
            current_model.load_state_dict(torch.load('./best_models/' + args.model + "_" + args.baseline + '.pth'))
        except:
            a = "do nothing"

    optimizer = torch.optim.SGD(current_model.parameters(), lr=args.lr)
    loss_func = torch.nn.CrossEntropyLoss()
    logging.disable(logging.NOTSET)

    best_dev_performance = 0
    for i in range(args.epochs):
        print("========================= epoch " + str(i) + " ==========================")
        train(current_model, tr_data, args, loss_func, optimizer) # save a few checkpoints and pick the best
        acc, p, r, f1, label_seq, pred_seq = test(current_model, val_data, args, loss_func, optimizer)
        print(" Dev Acc, P, R, F1: {}, {}, {}, {}".format(acc, p, r, f1))
        all_devs.append([acc, p, r, f1, label_seq, pred_seq])
        if f1 > best_dev_performance:
            print("New Best F1!!! " + str(f1))
            best_dev_performance = f1
            torch.save(current_model.state_dict(), './best_models_continue/' + args.model + "_" + args.baseline + '.pth')

    current_model.load_state_dict(torch.load('./best_models_continue/' + args.model + "_" + args.baseline + '.pth'))

    acc, p, r, f1, label_seq, pred_seq = test(current_model, test_data, args, loss_func, optimizer)
    all_devs.append([acc, p, r, f1, label_seq, pred_seq])

    with open('./best_models_continue/' + args.model + "_" + args.baseline + '.json', "w", encoding="UTF-8") as f:
        json.dump(all_devs, f)


    print(" Test Acc, P, R, F1: {}, {}, {}, {}".format(acc, p, r, f1))
    return [acc, p, r, f1, label_seq, pred_seq]



parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--gpu", default='0', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--seed", default=2020, type=int, required=False,
                    help="default random seed")
parser.add_argument("--model", default='bert-base-uncased', type=str, required=False,
                    help="choose the model to test")
parser.add_argument("--lr", default=0.0005, type=float, required=False,
                    help="initial learning rate")
parser.add_argument("--epochs", default=5, type=int, required=False,
                    help="number of training epochs")
parser.add_argument("--lrdecay", default=0.8, type=float, required=False,
                    help="learning rate decay every 5 epochs")
parser.add_argument("--max_len", default=128, type=int, required=False,
                    help="number of words")
parser.add_argument("--baseline", default="full", type=str, required=False,
                    help="the baseline to test") # full, sbert, no_debater, no_argument, no_topic, one_hot
parser.add_argument("--max_sent_num", default=5, type=int, required=False,
                    help="max number of sentences to process")
parser.add_argument("--continue_train", default=False, type=bool, required=False,
                    help="train with the previous best model")


args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device:', device)
n_gpu = torch.cuda.device_count()
print('number of gpu:', n_gpu)
torch.cuda.get_device_name(0)

# set your data path here
if args.baseline == "one_hot":
    data_class = DataLoader('./split.json', args)
    # torch.save(data_class, './data_file.pt') 

else:
    print("Loading Cached Weights...")
    data_class = torch.load('./data_file.pt')

# testing
# [acc, p, r, f1, label_seq, pred_seq]
answers = train_test_each(args, data_class.tensor_split)

# io
print('end')
