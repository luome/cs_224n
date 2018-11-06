import re
import random 
import torch
import gensim 
import numpy as np


# a lambda funtion to flatten the sequence
flatten = lambda l:[item for sublist in l for item in sublist]


def load_data(training=False):
    if training:
        data = open('data/train_5500.label.txt', 'r', encoding='latin-1').readlines()
    else:
        data = open('data/TREC_10.label.txt', 'r', encoding='latin-1').readlines()
        
    data = [[d.split(':')[1][:-1], d.split(':')[0]] for d in data] 
    X, y = list(zip(*data))   
    X = list(X)
    for i, x in enumerate(X):
        X[i] = re.sub(r'\d', '#', x).split()[1:-1]
    return X, y


def build_vocabulary(X, y):
    vocab = list(flatten(X))
    word2index = {'<PAD>': 0, '<UNK>':1}
    for v in vocab:
        if word2index.get(v) is None:
            word2index[v] = len(word2index)
    
    target2index = {}
    for cl in list(y):
        if target2index.get(cl) is None:
            target2index[cl] = len(target2index)
    
    return word2index, target2index


def get_batch(batch_size, train_data, shuffle=True):
    if shuffle:
        random.shuffle(train_data)
    batch_num = int((len(train_data)-1)/batch_size) + 1
    for batch_num in range(batch_num):
        start_index = batch_size * batch_num
        end_index = min((batch_num+1)*batch_size, len(train_data))
        yield train_data[start_index: end_index]


def pad_to_batch(batch, word2index):
    x, y = zip(*batch)
    max_x = max([s.size(1) for s in x])
    x_p = [] # the padded x
    for i in range(len(batch)):
        x_size = x[i].size(1)
        if x_size < max_x:
            x_p.append(torch.cat([x[i], torch.tensor([word2index['<PAD>']] * (max_x - x_size), dtype=torch.long).view(1, -1)], 1))
        else:
            x_p.append(x[i])
    return torch.cat(x_p), torch.cat(y).view(-1)


def prepare_sequence(seq, to_index):
    index = list(map(lambda w:to_index[w] if to_index.get(w) is not None else to_index['UNK'], seq))
    return torch.tensor(index, dtype=torch.long)


def load_word_vector(word2index):
    model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    pretrained = []
    for key in word2index.keys():
        try:
            pretrained.append(model[word2index[key]])
        except:
            pretrained.append(np.random.randn(300))
    
    pretrained_vectors = np.vstack(pretrained)
    return pretrained_vectors


def preprocess():
    X_train, y_train = load_data(training=True)
    X_test, y_test = load_data()
    X = flatten([X_train, X_test])
    y = flatten([y_train, y_test])
    word2index, target2index = build_vocabulary(X, y)
    # index2word = {v:k for k,v in word2index.items()}
    # index2target = {v:k for k,v in target2index.items()}
    X_p, y_p = [], []
    for pair in zip(X, y):
        X_p.append(prepare_sequence(pair[0],word2index).view(1, -1))
        y_p.append(torch.tensor([target2index[pair[1]]], dtype=torch.long).view(1, -1))
    
    data_p = list(zip(X_p, y_p))
    train_data = data_p[:len(X_train)]
    test_data = data_p[len(X_train):]
    return train_data, test_data, word2index, target2index 
    

