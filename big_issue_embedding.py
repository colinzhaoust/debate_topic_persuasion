# -*- coding: utf-8 -*-
# imports needed and logging

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gzip
import gensim
import logging
import json
import os
import random
import numpy as np
import math
from gensim.models import Word2Vec
# import torch
# from sklearn.decomposition import PCA
# from matplotlib import pyplot
# from torch import nn
# from torch.nn import init
# from torch import optim
import time
from tqdm import tqdm_notebook as tqdm


def big_issue_embedding():
    with open("./data/users.json", "r") as f:
        users = json.load(f)

    sentences = []

    for user in users.values():
        user_pro = []
        user_con = []
        user_und = []
        user_no = []
        # print(user["big_issues_dict"])
        for key in user["big_issues_dict"].keys():
            if user["big_issues_dict"][key] == "Pro":
                user_pro.append(key)
            if user["big_issues_dict"][key] == "Con":
                user_con.append(key)
            if user["big_issues_dict"][key] == "Und":
                user_und.append(key)
            if user["big_issues_dict"][key] == "N/O":
                user_no.append(key)

        if len(user_pro) > 0:
            sentences.append(user_pro)
        if len(user_con) > 0:
            sentences.append(user_con)
        if len(user_und) > 0:
            sentences.append(user_und)
        if len(user_no) > 0:
            sentences.append(user_no)

    # print(len(sentences))

    model = Word2Vec(sentences, size=10, window=30)
    # print(model)

    words = list(model.wv.vocab)

    print("Full list of big issues.")
    print(words)
    # print(model["Abortion"])

    # prepare similarity matrix
    issue_sim_dic = dict()
    embedding_dic = dict()
    
    for word in words:
        issue_sim_dic[word] = model.wv.most_similar_cosmul(positive=[word], topn=len(words)-1)
        embedding_dic[word] = model[word].tolist()


    return model, words, issue_sim_dic, embedding_dic

if __name__ == "__main__":
    model, words, issue_sim_dic, embedding_dic = big_issue_embedding()
    print(issue_sim_dic)

    with open("big_issue_embedding.json","w",encoding="UTF-8") as f:
        json.dump(embedding_dic, f)
        
    with open("issue_similarity.json","w",encoding="UTF-8") as f:
        json.dump(issue_sim_dic, f)

# from sklearn.decomposition import PCA
# from matplotlib import pyplot

# # replace this line with a list of vectors
# X = model[model.wv.vocab]

# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# pyplot.scatter(result[:, 0], result[:, 1])

# for i, word in enumerate(words):
#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

# pyplot.show()
