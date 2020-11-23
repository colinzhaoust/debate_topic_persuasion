import ujson as json
import torch
# from pytorch_transformers import *
import logging
import argparse
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
import math
import numpy
import collections
import random
from tqdm import tqdm
# from pytorch_transformers import *
from transformers import *

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# tools
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from arglex.Classifier import Classifier

# load pretrained embedding
from big_issue_embedding import big_issue_embedding
from user_aspect_embedding import user_attritbute_embedding

# argument lexicon
arglex = Classifier()

# big issues 
BIG_ISSUES = ['Abortion', 'Affirmative Action', 'Animal Rights', 'Barack Obama', 'Border Fence', 'Capitalism', 'Civil Unions', 'Death Penalty', 'Drug Legalization', 'Electoral College', 'Environmental Protection', 'Estate Tax', 'European Union', 'Euthanasia', 'Federal Reserve', 'Flat Tax', 'Free Trade', 'Gay Marriage', 'Global Warming Exists', 'Globalization', 'Gold Standard', 'Gun Rights', 'Homeschooling', 'Internet Censorship', 'Iran-Iraq War', 'Labor Union', 'Legalized Prostitution', 'Medicaid & Medicare', 'Medical Marijuana', 'Military Intervention', 'Minimum Wage', 'National Health Care', 'National Retail Sales Tax', 'Occupy Movement', 'Progressive Tax', 'Racial Profiling', 'Redistribution', 'Smoking Ban', 'Social Programs', 'Social Security', 'Socialism', 'Stimulus Spending', 'Term Limits', 'Torture', 'United Nations', 'War in Afghanistan', 'War on Terror', 'Welfare']
USEFUL_CATS = ['political_ideology', 'education', 'ethnicity', 'interested', 'gender' , 'religious_ideology']
TARGET_LABEL = ["Pro","Con"]

def linguistic_feature_generator(sent_list, feature_set=["len","sub-polar","arglex"]):
    
    linguistic_vec = []
    for sent in sent_list:
        if sent[-1] in "abcdefghijklmnopqrstuvwxyz":
            sent += "."

    sep = " "
    text = sep.join(sent_list)

    avg_length = 0
    avg_sub = 0
    avg_polar = 0
    count = len(sent_list)

    # failed situtaion
    if count == 0:
        return [0.0 for i in range(8)]

    # sentiment
    sent_blob = TextBlob(text)
    average_sub = sent_blob.sentiment.subjectivity
    average_polar = sent_blob.sentiment.polarity

    # len
    for sent in sent_list:
        avg_length += len(word_tokenize(sent))

    avg_length = avg_length/count

    # arg
    lexicon_score = arglex.analyse(text)
    # in the original arglex, we have
    # ['0-Assessments', '1-Authority', '2-Causation', '3-Conditionals', '4-Contrast', '5-Difficulty', '6-Doubt', '7-Emphasis',\
    #     '8-Generalization', '9-Inconsistency', '10-Inyourshoes', '11-Necessity', '12-Possibility', '13-Priority', '14-Rhetoricalquestion',\
    #    '15-Structure', '16-Wants']
    # Here we only care about Authority, Conditionals, Contrast, Difficulty, Necessity
    lex_vec = [lexicon_score[1], lexicon_score[3], lexicon_score[4], lexicon_score[5], lexicon_score[11]]
    
    if "len" in feature_set:
        linguistic_vec.append(avg_length)
    if "sub-polar" in feature_set:
        linguistic_vec.append(average_sub)
        linguistic_vec.append(average_polar)
    if "arglex" in feature_set:
        linguistic_vec.extend(lex_vec)

    return linguistic_vec


class BertEncoder(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.lm = BertModel(config)
        # self.embedding_size = 300

    def forward(self, sents):
        # forwarding the sents and use the average embedding as the results

        representation  = self.lm(sents) #.unsqueeze(0)) # num_sent * sent_len * emb
        # print(representation[0].size)
        sent_representation = torch.mean(representation[0], dim=1) # num_sent * emb
        # print(sent_representation.size)
        overall_representation = torch.mean(sent_representation, dim=0) # 1 *  emb

        return overall_representation

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


class DebateModel(torch.nn.Module):

    def __init__(self, num_task, config):
        super(DebateModel, self).__init__()
        
        self.num_task = num_task
        self.bert = BertEncoder.from_pretrained(config)
        # self.lstm = LSTMEncoder(300,300)

        self.text_specific = torch.nn.Linear(1024, 128)
        self.ling_specific = torch.nn.Linear(8, 5)
        self.cat_specific = torch.nn.Linear(len(USEFUL_CATS) * 10, 64) # 6 * 10
        self.task_specific = torch.nn.Linear(10, 5)

        self.embedding_size = 300
        self.hidden_dim = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.classification = torch.nn.Linear(128+64+5, self.hidden_dim) # remaining
        self.second_last_layer = torch.nn.Linear(self.hidden_dim,self.hidden_dim)
        self.last_layers = self.init_per_task_last_layers(self.num_task)
    
    def init_per_task_last_layers(self, num_task):
        # we only cares about Pro/Con here
        return torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim + 5, len(TARGET_LABEL)) for i in range(num_task)])

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, cat_vec, ling_vec, tokenized_sents, task_embs):
        cat_rep = self.cat_specific(cat_vec)
        ling_rep = self.ling_specific(ling_vec)
        
        text_rep = self.bert(tokenized_sents)
        text_rep = self.text_specific(text_rep)

        cls_rep = torch.cat([cat_rep, ling_rep, text_rep])
        # two layers NN for the classification module
        cls_rep = self.classification(cls_rep)
        cls_rep = self.second_last_layer(cls_rep)

        task_results = []
        for i,task_emb in enumerate(task_embs):
            task_rep = self.task_specific(task_emb)
  
            combined_rep = torch.cat([cls_rep, task_rep]).to(device)

            task_result = self.last_layers[i](combined_rep)
            task_results.append(task_result)

        # predictions = torch.cat(task_results)

        return task_results


class DataLoader:
    def __init__(self, data_path, args):

        self.args = args
        self.users = self.initial_loading(data_path)
        print("Successfully load the user data")

        self.tokenizer = BertTokenizer.from_pretrained(args.model)
        # self.word_embeddings = self.load_embedding_dict('glove.txt')

        # Calculate as loading
        # _, self.big_issues, self.issue_sim_dic, self.issue_emb_dic = big_issue_embedding()
        # _, _, self.att_emb_dic = user_attritbute_embedding()

        # Load pretrained
        self.issue_emb_dic, self.issue_sim_dic, self.att_emb_dic = self.load_pretrained_embedding_dict()

        print("Successfully load all the embeddings")

        print("Start processing the data.")
        self.data_by_issue = self.tensorize_examples(self.users)
        self.data_collection =  self.seperate_tr_val_test(self.data_by_issue)
        print("Successfully processed the data.")

    def initial_loading(self, data_path):
        with open(data_path, "r",encoding="UTF-8") as f:
            users = json.load(f)

        return users

    def load_pretrained_embedding_dict(self):

        with open("big_issue_embedding.json","r",encoding="UTF-8") as f:
            issue_emb_dic = json.load(f)

        with open("issue_similarity.json","r",encoding="UTF-8") as f:
            issue_sim_dic = json.load(f)

        with open("user_attritbute_embedding.json","r",encoding="UTF-8") as f:
            att_emb_dic = json.load(f)

        return issue_emb_dic, issue_sim_dic, att_emb_dic

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

    def tensorize_examples(self, users):
        data_by_issue = dict()

        for issue in BIG_ISSUES:
            data_by_issue[issue] = dict()
            # get the auxiliary issues, [issue,similarity]
            data_by_issue[issue]["aux_issues"] = self.issue_sim_dic[issue][:(self.args.num_task-1)]
            data_by_issue[issue]["users"] = []

        for user in tqdm(users[:100]):
            tensorized_user = dict()

            # categorical attributes
            cat_one_hot = []
            cat_emb = []
            for cat in USEFUL_CATS:
                cat_emb.extend(self.att_emb_dic[cat + ":" +user[cat]])

                cat_id = user[cat+"_id"]
                cat_num = user[cat+"_len"]
                one_hot = [0.0 for i in range(cat_num)]
                one_hot[cat_id] = 1.0
                cat_one_hot.extend(one_hot)

            text_sents = []
            for opinion in user["opinions"]:
                text_sents.extend(sent_tokenize(opinion))
            for debate in user["debates"]:
                text_sents.extend(sent_tokenize(debate))

            # linguistic vectors
            linguistic_vector = linguistic_feature_generator(text_sents, feature_set=["len","sub-polar","arglex"])

            # tokenized sentences for BERT
            tokenized_sents = []
            for sent_num, sent in enumerate(text_sents):
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
                tokenized_sents.append(tokenized_sent)

            # matching the issues into the tasks
            targets = []
            for issue in BIG_ISSUES:
                # If anyone in main or the auxiliary issues has a label out of the desired ones
                out_of_scope = 0
                auxes = self.issue_sim_dic[issue][:(self.args.num_task-1)]

                if user["big_issues_dict"][issue] not in TARGET_LABEL:
                    out_of_scope = 1
                for aux in auxes:
                    if user["big_issues_dict"][aux[0]] not in TARGET_LABEL:
                        out_of_scope = 1

                if out_of_scope == 0:
                    tmp = []
                    tmp.append([issue,1.0])
                    for aux in auxes:
                        tmp.append(list(aux))
                    targets.append(tmp)                

            for target in targets:
                # target [[main, 1], [aux,sim]]
                label_seq = []
                task_embs = []
                sim_seq = []
                for issue_sim in target:
                    # TARGET_LABEL = [Pro, Con]
                    label_seq.append(TARGET_LABEL.index(user["big_issues_dict"][issue_sim[0]]))
                    task_embs.append(self.issue_emb_dic[issue_sim[0]])
                    sim_seq.append(issue_sim[1])
                # name of the main issue
                data_by_issue[target[0][0]]["users"].append(
                {'cat_one_hot':torch.tensor(cat_one_hot).to(device),
                    'cat_emb': torch.tensor(cat_emb).to(device),
                    'ling_vec':torch.tensor(linguistic_vector).to(device),
                    'tokenized_sents': torch.tensor(tokenized_sents).to(device),
                    'label_seq': torch.tensor(label_seq).to(device),
                    'task_embs': torch.tensor(task_embs).to(device),
                    'sim_seq': torch.tensor(sim_seq).to(device)
                    })

        return data_by_issue

    def seperate_tr_val_test(self, data_by_issue):
        seperated_data_collection = dict()
        count = 0
        portions=[0.7,0.85,1]

        for issue in BIG_ISSUES:
            collected_users = data_by_issue[issue]["users"]
            random.seed(self.args.seed)
            # random.shuffle(collected_users)
            tr_data = [] 
            val_data = []
            test_data = []

            count +=  len(collected_users)

            for i, user in enumerate(collected_users):
                if i < int(portions[0]*len(collected_users)):
                    tr_data.append(user)
                elif i >= int(portions[0]*len(collected_users)) and i < int(portions[1]*len(collected_users)):
                    val_data.append(user)
                else:
                    test_data.append(user)

            seperated_data_collection[issue] = dict()
            seperated_data_collection[issue]["train"] = tr_data
            seperated_data_collection[issue]["val"] = val_data
            seperated_data_collection[issue]["test"] = test_data

        print("Successfully load "+ str(count)+ " pairs of user-attitude.")

        return seperated_data_collection


def train(model, data, args, loss_func, optimizer):
    all_loss = 0
    print('training:')
    random.shuffle(data)
    model.train()
    selected_data = data
    # tmp_example
        # {'cat_one_hot':torch.tensor(cat_one_hot).to(device),
        #     'cat_emb': torch.tensor(cat_emb).to(device),
        #     'ling_vec':torch.tensor(linguistic_vector).to(device),
        #     'tokenized_sents': torch.tensor(bert_tokenized_sent2).to(device),
        #     'label_seq': torch.tensor([int(label)]).to(device),
        #     'task_embs': torch.tensor([int(label)]).to(device),
        #     'sim_seq': torch.tensor([int(label)]).to(device)
        #     })

    for tmp in tqdm(selected_data):
        task_results = model(cat_vec=tmp['cat_emb'].to(device), ling_vec=tmp['ling_vec'].to(device), tokenized_sents=tmp['tokenized_sents'].to(device), task_embs=tmp['task_embs'].to(device))
        for i, task_prediction in enumerate(task_results):
            # here we assigned the weighted sum
            if i == 0: # main task: 1 * loss
                loss = loss_func(task_prediction.view(1, -1), tmp['label_seq'][i].unsqueeze(0))
            else: # auxiliarys: sim * loss
                loss += tmp["sim_seq"][i] * loss_func(task_prediction.view(1, -1), tmp['label_seq'][i].unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_loss += loss.item()

    print('current loss:', all_loss / len(data))


def test(model, data, args, loss_func, optimizer):
    correct_count = 0
    ans_seq = []
    # print('Testing:')
    model.eval()
    for tmp in tqdm(data):
        task_results = model(cat_vec=tmp['cat_emb'].to(device), ling_vec=tmp['ling_vec'].to(device), tokenized_sents=tmp['tokenized_sents'].to(device), task_embs=tmp['task_embs'].to(device))
        main_prediction =  task_results[0]

        if tmp['label_seq'][0].data[0] == 1:
            # current example is positive
            if main_prediction.data[1] >= main_prediction.data[0]:
                correct_count += 1
                ans_seq.append(1)
            else:
                ans_seq.append(0)
        else:
            # current example is negative
            if main_prediction.data[1] <= main_prediction.data[0]:
                correct_count += 1
                ans_seq.append(1)
            else:
                ans_seq.append(0)
    # print('current accuracy:', correct_count, '/', len(data), correct_count / len(data))
    return correct_count / len(data), ans_seq


def train_test_by_issue(args, data_collection):
    # correct_count = dict()
    all_count = dict()
    # correct_count['overall'] = []
    all_count['overall'] = []
    print("Start training...")

    for i, issue in enumerate(BIG_ISSUES):
        print("We are working on the Issue "+ str(i)+": " + issue)

        # correct_count[issue] = []
        all_count[issue] = []
        data_here = data_collection[issue]
        tr_data, val_data, test_data = data_here["train"], data_here["val"], data_here["test"]

        print("Loading the full model...")
        logging.disable(logging.CRITICAL)
        # initialize model
        current_model = DebateModel(args.num_task,args.model)

        current_model.to(device)
        optimizer = torch.optim.SGD(current_model.parameters(), lr=args.lr)
        loss_func = torch.nn.CrossEntropyLoss()
        logging.disable(logging.NOTSET)

        best_dev_performance = 0
        for i in range(args.epochs):
            print("========================= epoch " + str(i) + " ==========================")
            train(current_model, tr_data, args, loss_func, optimizer) # save a few checkpoints and pick the best
            type_acc, _ = test(current_model, val_data, args, loss_func, optimizer)
            print(" Dev Acc: ")
            if type_acc > best_dev_performance:
                print("New Best Acc!!!" + str(type_acc))
                best_dev_performance = type_acc
                torch.save(current_model.state_dict(), './best_models/' + args.model + '.pth')

        current_model.load_state_dict(torch.load('./best_models/' + args.model + '.pth'))
        type_acc, type_ans_seq = test(current_model, val_data, args, loss_func, optimizer)

        print("Test Accurracy for " + issue + ": " + str(type_acc))
        print()
        print()
        all_count[issue].extend(type_ans_seq)
        all_count['overall'].extend(type_ans_seq)

    types = ["overall"]
    types.extend(BIG_ISSUES)
    accuracy_by_type = dict()

    for this_type in types:
        accuracy_by_type[this_type] = np.mean(np.array(all_count[this_type]))

    return accuracy_by_type



parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--gpu", default='0', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--seed", default=2020, type=int, required=False,
                    help="default random seed")
parser.add_argument("--model", default='bert-large-uncased', type=str, required=False,
                    help="choose the model to test")
parser.add_argument("--lr", default=0.0001, type=float, required=False,
                    help="initial learning rate")
parser.add_argument("--epochs", default=5, type=int, required=False,
                    help="number of training epochs")
parser.add_argument("--lrdecay", default=0.8, type=float, required=False,
                    help="learning rate decay every 5 epochs")
parser.add_argument("--method", default="r", type=str, required=False,
                    help="wsc or r, two kinds of settings")
parser.add_argument("--fold", default=0, type=int, required=False,
                    help="testing which fold")
parser.add_argument("--num_task", default=2, type=int, required=False,
                    help="number of predicted tasks")
parser.add_argument("--max_len", default=128, type=int, required=False,
                    help="number of words")
parser.add_argument("--baseline", default="full", type=str, required=False,
                    help="the baseline to test")
parser.add_argument("--max_sent_num", default=5, type=int, required=False,
                    help="max number of sentences to process")


args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device:', device)
n_gpu = torch.cuda.device_count()
print('number of gpu:', n_gpu)
torch.cuda.get_device_name(0)

# set your data path here
# data_class = DataLoader('./text_debate_users.json', args)
# torch.save(data_class, './data_file.pt') 

# print("Loading Cached Weights...")
data_class = torch.load('./data_file.pt')

# testing
# args.epochs = 2 
accuracy_by_type = train_test_by_issue(args, data_class.data_collection)

# IO
print("With " + str(args.num_task) + ", the overall accuracy is: " + str(accuracy_by_type["overall"]))

print('end')
