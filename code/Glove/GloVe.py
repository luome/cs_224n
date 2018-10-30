import torch
import torch.nn as nn
import torch.optim as optim
import random 
import nltk
from collections import Counter
from itertools import combinations_with_replacement
import numpy as np 
import torch.nn.functional as F

# This script simplely implements the GloVe algorithm using a nltk corpus
# w.r.t the git repository from DSKSD (see readme) 


# use cuda 
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

def get_batch(batch_size, train_data):
    random.shuffle(train_data)
    start_index = 0 
    end_index = batch_size 
    while end_index < len(train_data):
        batch = train_data[start_index:end_index]
        temp = end_index 
        end_index = end_index + batch_size
        start_index = temp
        yield batch
    
    if end_index >=  len(train_data):
        batch = train_data[start_index:]
        yield batch


# a helper funtion to put all items in one list 
flatten = lambda l:[item for sublist in l for item in sublist]

# load the corpus
corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:500]
corpus = [[word.lower() for word in sent] for sent in corpus]

# build the vocabulary
vocab = list(set(flatten(corpus)))
word2index = {}
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
word2index['<UNK>'] = len(word2index)

index2word = {v:k for k, v in word2index.items()}

# n-gram windows
WINDOW_SIZE = 5
windows = flatten([list(nltk.ngrams(['<DUMMY>']*WINDOW_SIZE + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in corpus])
window_data = []
for window in windows:
        for i in range(WINDOW_SIZE*2 + 1):
            if i == WINDOW_SIZE or window[i] == '<DUMMY>':
                continue
            window_data.append((window[WINDOW_SIZE],window[i]))


# the weighting function f(x)
def weighting(w_i, w_j):
    try:
        x_ij = X_ik[(w_i, w_j)]
    except:
        x_ij = 1

    # the following two parameters is recommended by the artice
    x_max = 100
    alpha = 0.75 

    if x_ij < x_max:
        result = (x_ij/x_max) **alpha
    else:
        result = 1
    
    return result


# prepare the data
X_i = Counter(flatten(corpus))
X_ik_window_5 = Counter(window_data)
X_ik = {}
weighting_dic = {}
for bigram in combinations_with_replacement(vocab, 2):
    # count the number of times any word k appears in the context of word i
    if X_ik_window_5.get(bigram) is not None:
        co_occurrence = X_ik_window_5[bigram]
        X_ik[bigram] = co_occurrence + 1
        X_ik[(bigram[1], bigram[0])] = co_occurrence + 1
    else:
        pass
    # compute the weighting function
    weighting_dic[bigram] = weighting(bigram[0], bigram[1])
    weighting_dic[(bigram[1], bigram[0])] = weighting(bigram[1], bigram[0])

# test = random.choice(window_data)
# print(test)
# try:
#     print(X_ik[test[0], test[1]] == X_ik[(test[1], test[0])])
# except:
#     1

# helper functions 
def prepare_sequence(seq, word2index):
    index = list(map(lambda w:word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return torch.tensor(index, dtype=torch.long)

def prepare_word(word, word2index):
    return torch.tensor([word2index[word]], dtype=torch.long) if word2index.get(word) is not None else torch.tensor([word2index["<UNK>"]])


# prepare train data
u_p = []  # center word vector
v_p = []  # context word vector 
co_p = []  # log(x_ij) 
weight_p = []  # f(X_ij) 

for pair in window_data:
    u_p.append(prepare_word(pair[0], word2index).view(1, -1))
    v_p.append(prepare_word(pair[1], word2index).view(1, -1))

    try:
        cooc = X_ik[pair]
    except:
        cooc = 1
    
    co_p.append(torch.log(torch.tensor([cooc], dtype=torch.float)).view(1, -1))
    weight_p.append(torch.tensor([weighting_dic[pair]], dtype=torch.float).view(1, -1))

train_data = list(zip(u_p, v_p, co_p, weight_p))
del u_p
del v_p
del co_p
del weight_p
# print(train_data[0])

# the model
class GloVe(nn.Module):
    def __init__(self, vocab_size, projection_dim):
        super(GloVe, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim)
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)

        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)

        init_range = (2./(vocab_size + projection_dim))**0.5
        self.embedding_v.weight.data.uniform_(-init_range, init_range)
        self.embedding_u.weight.data.uniform_(-init_range, init_range)
        self.v_bias.weight.data.uniform_(-init_range, init_range)
        self.u_bias.weight.data.uniform_(-init_range, init_range)

    def forward(self, center_words, target_words, coocs, weights):
        center_embeds = self.embedding_v(center_words)
        target_embeds = self.embedding_u(target_words)
        center_bias = self.v_bias(center_words).squeeze(1)
        target_bias = self.u_bias(target_words).squeeze(1)

        inner_product = target_embeds.bmm(center_embeds.transpose(1,2)).squeeze(2)

        loss = weights * torch.pow((inner_product + center_bias + target_bias - coocs), 2)

        return torch.sum(loss)
    
    def prediction(self, inputs):
        v_embeds = self.embedding_v(inputs)
        u_embeds = self.embedding_u(inputs)

        return v_embeds + u_embeds

EMBEDDING_SIZE = 50
BATCH_SIZE = 256
EPOCH = 50

losses = []
model = GloVe(len(word2index), EMBEDDING_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# trainning
for epoch in range(EPOCH):
    for i, batch in enumerate(get_batch(BATCH_SIZE, train_data)):

        inputs, targets, coocs, weights = zip(*batch)
        
        inputs = torch.cat(inputs).to(device)
        targets = torch.cat(targets).to(device)
        coocs = torch.cat(coocs).to(device)
        weights = torch.cat(weights).to(device)

        model.zero_grad()
        
        loss = model(inputs, targets, coocs, weights)

        loss.backward()

        optimizer.step()

        losses.append(loss.data.tolist())

    if epoch % 10 == 0:
        print("Epoch:%d, mean_loss:%.02f" %(epoch, np.mean(losses)))
        losses = []
    
# Test 
def word_similarity(target, vocab):
    target_v = model.prediction(prepare_sequence(target, word2index).to(device))
    similarities = []
    for i in range(len(vocab)):
        if vocab[i] == target:
            continue
        vector = model.prediction(prepare_word(list(vocab)[i], word2index).to(device))

        cosine_sim = F.cosine_similarity(target_v, vector).data.tolist()[0]

        similarities.append([vocab[i], cosine_sim])
    
    return sorted(similarities, key=lambda x:x[1], reverse=True)[:10]

test = random.choice(list(vocab))

print('test word:', test)
print('words closest to the test word:', word_similarity(test, vocab))



