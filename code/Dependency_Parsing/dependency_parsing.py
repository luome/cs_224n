import nltk
from nltk.draw.util import CanvasFrame
from nltk.tree import Tree
from nltk.draw import TreeWidget
import os
from tkinter import mainloop
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

# use cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


class TransitionState(object):
    def __init__(self, tagged_sent):
        self.root = ('ROOT', '<root>', -1)
        self.stack = [self.root]
        self.buffer = [(s[0], s[1], i) for i, s in enumerate(tagged_sent)]
        self.address = [s[0] for s in tagged_sent] + [self.root[0]]
        self.arcs = []
        self.terminal = False

    def __str__(self):
        return 'stack:%s \nbuffer:%s' % (str([s[0] for s in self.stack]), str([b[0] for b in self.buffer]))

    def shift(self):
        if len(self.buffer) >= 1:
            self.stack.append(self.buffer.pop(0))
        else:
            print("Empty buffer.")

    def left_arc(self, relation=None):
        if len(self.stack) >= 2:
            arc = {}
            s2 = self.stack[-2]
            s1 = self.stack[-1]
            arc['graph_id'] = len(self.arcs)
            arc['form'] = s1[0]
            arc['addr'] = s1[2]
            arc['head'] = s2[2]
            arc['pos'] = s1[1]
            if relation:
                arc['relation'] = relation
            self.arcs.append(arc)
            self.stack.pop(-2)
        elif self.stack == [self.root]:
            print("Element lacking.")

    def right_arc(self, relation=None):
        if len(self.stack) >= 2:
            arc = {}
            s2 = self.stack[-2]
            s1 = self.stack[-1]
            arc['graph_id'] = len(self.arcs)
            arc['form'] = s2[0]
            arc['addr'] = s2[2]
            arc['head'] = s1[2]
            arc['pos'] = s2[1]
            if relation:
                arc['relation'] = relation
            self.arcs.append(arc)
            self.stack.pop(-1)
        elif self.stack == [self.root]:
            print("Element lacking.")

    def get_left_most(self, index):
        left = ['<NULL>', '<NULL>', None]
        if index == None:
            return left
        for arc in self.arcs:
            if arc['head'] == index:
                left = [arc['form'], arc['pos'], arc['addr']]
                break
        return left

    def get_right_most(self, index):
        right = ['<NULL>', '<NULL>', None]
        if index == None:
            return right
        for arc in reversed(self.arcs):
            if arc['head'] == index:
                right = [arc['form'], arc['pos'], arc['addr']]
                break
        return right

    def is_done(self):
        return len(self.buffer) == 0 and self.stack == [self.root]

    def to_tree_string(self):
        if self.is_done() == False:
            return None
        ingredient = []
        for arc in self.arcs:
            ingredient.append([arc['form'], self.address[arc['head']]])
        ingredient = ingredient[-1:] + ingredient[:-1]
        return self._make_tree(ingredient, 0)

    def _make_tree(self, ingredient, i, new=True):
        if new:
            treestr = "("
            treestr += ingredient[i][0]
            treestr += " "
        else:
            treestr = ""
        ingredient[i][0] = "CHECK"

        parents, _ = list(zip(*ingredient))

        if ingredient[i][1] not in parents:
            treestr += ingredient[i][1]
            return treestr

        else:
            treestr += "("
            treestr += ingredient[i][1]
            treestr += " "
            for node_i, node in enumerate(parents):
                if node == ingredient[i][1]:
                    treestr += self._make_tree(ingredient, node_i, False)
            treestr = treestr.strip()
            treestr += ")"

        if new:
            treestr += ")"
        return treestr


def draw_nltk_tree(tree):
    cf = CanvasFrame()
    tc = TreeWidget(cf.canvas(), tree)
    tc['node_font'] = 'arial 15 bold'
    tc['leaf_font'] = 'arial 15'
    tc['node_color'] = '#005990'
    tc['leaf_color'] = '#3F8F57'
    tc['line_color'] = '#175252'
    cf.add_widget(tc, 50, 50)
    cf.canvas().pack()
    mainloop()  # use tkinter to show the tree image


# if __name__ == '__main__':
#     state = TransitionState(nltk.pos_tag('He has good control .'.split()))
#     state.shift()
#     state.shift()
#     state.left_arc()
#     state.shift()
#     state.shift()
#     state.left_arc()
#     state.right_arc()
#     state.shift()
#     state.right_arc()
#     state.right_arc()
#     print(state.is_done())
#     print(state.to_tree_string())
#     draw_nltk_tree(Tree.fromstring(state.to_tree_string()))

def prepare_sequence(seq, to_index):
    index = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index['<NULL>'], seq))
    return torch.tensor(index, dtype=torch.long)


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

    if end_index >= len(train_data):
        batch = train_data[start_index:]
        yield batch


def get_feature(transition_state, word2index, tag2index, label2index=None):
    s = []  # word features s1, s2, s3, b1, b2, b3
    t = []  # tag features st1, st2, st3, bt1, bt2, bt3

    s.append(transition_state.stack[-1][0]) if len(transition_state.stack) >= 1 and transition_state.stack[-1][
        0] in word2index.keys() else s.append('<NULL>')
    s.append(transition_state.stack[-2][0]) if len(transition_state.stack) >= 2 and transition_state.stack[-2][
        0] in word2index.keys() else s.append('<NULL>')
    s.append(transition_state.stack[-3][0]) if len(transition_state.stack) >= 3 and transition_state.stack[-3][
        0] in word2index.keys() else s.append('<NULL>')

    t.append(transition_state.stack[-1][1]) if len(transition_state.stack) >= 1 and transition_state.stack[-1][
        1] in tag2index.keys() else t.append('<NULL>')
    t.append(transition_state.stack[-2][1]) if len(transition_state.stack) >= 2 and transition_state.stack[-2][
        1] in tag2index.keys() else t.append('<NULL>')
    t.append(transition_state.stack[-3][1]) if len(transition_state.stack) >= 3 and transition_state.stack[-3][
        1] in tag2index.keys() else t.append('<NULL>')

    s.append(transition_state.buffer[0][0]) if len(transition_state.buffer) >= 1 and transition_state.buffer[0][
        0] in word2index.keys() else s.append('<NULL>')
    s.append(transition_state.buffer[1][0]) if len(transition_state.buffer) >= 2 and transition_state.buffer[1][
        0] in word2index.keys() else s.append('<NULL>')
    s.append(transition_state.buffer[2][0]) if len(transition_state.buffer) >= 3 and transition_state.buffer[2][
        0] in word2index.keys() else s.append('<NULL>')

    t.append(transition_state.buffer[0][1]) if len(transition_state.buffer) >= 1 and transition_state.buffer[0][
        1] in tag2index.keys() else t.append('<NULL>')
    t.append(transition_state.buffer[1][1]) if len(transition_state.buffer) >= 2 and transition_state.buffer[1][
        1] in tag2index.keys() else t.append('<NULL>')
    t.append(transition_state.buffer[2][1]) if len(transition_state.buffer) >= 3 and transition_state.buffer[2][
        1] in tag2index.keys() else t.append('<NULL>')

    lc_s1 = transition_state.get_left_most(transition_state.stack[-1][2]) if len(
        transition_state.stack) >= 1 else transition_state.get_left_most(None)
    rc_s1 = transition_state.get_right_most(transition_state.stack[-1][2]) if len(
        transition_state.stack) >= 1 else transition_state.get_right_most(None)

    lc_s2 = transition_state.get_left_most(transition_state.stack[-2][2]) if len(
        transition_state.stack) >= 2 else transition_state.get_left_most(None)
    rc_s2 = transition_state.get_right_most(transition_state.stack[-2][2]) if len(
        transition_state.stack) >= 2 else transition_state.get_right_most(None)

    words, tags, _ = zip(*[lc_s1, rc_s1, lc_s2, rc_s2])
    s.extend(words)
    t.extend(tags)

    return prepare_sequence(s, word2index).view(1, -1), prepare_sequence(t, tag2index).view(1, -1)

# data from https://github.com/rguthrie3/DeepDependencyParsingProblemSet

flatten = lambda l: [item for sublist in l for item in sublist]
data = open('./data/train.txt', 'r').readlines()
vocab = open('./data/vocab.txt', 'r').readlines()

splited_data = [[nltk.pos_tag(d.split('|||')[0].split()), d.split('|||')[1][:-1].split()] for d in data]
train_x, train_y = list(zip(*splited_data))
train_x_f = flatten(train_x)
sents, pos_tags = list(zip(*train_x_f))

tag2index = {v: i for i, v in enumerate(set(pos_tags))}
tag2index['<root>'] = len(tag2index)
tag2index['<NULL>'] = len(tag2index)

vocab = [v.split('\t')[0] for v in vocab]
word2index = {v: i for i, v in enumerate(vocab)}
word2index['ROOT'] = len(word2index)
word2index['<NULL>'] = len(word2index)

actions = ['SHIFT', 'REDUCE_L', 'REDUCE_R']
action2index = {v: i for i, v in enumerate(actions)}

train_data = []

for tx, ty in splited_data:
    state = TransitionState(tx)
    transition = ty + ['REDUCE_R']  # root
    while len(transition):
        features = get_feature(state, word2index, tag2index)
        # print([tensor.shape for tensor in flatten(features)])
        action = transition.pop(0)
        actionTensor = torch.tensor([action2index[action]], dtype=torch.long).view(1, -1)
        train_data.append([features, actionTensor])
        if action == 'SHIFT':
            state.shift()
        elif action == 'REDUCE_R':
            state.right_arc()
        elif action == 'REDUCE_L':
            state.left_arc()

class NeuralDependencyParser(nn.Module):
    def __init__(self, w_size, w_embed_dim, t_size, t_embed_dim, hidden_size, target_size):
        super(NeuralDependencyParser, self).__init__()

        self.w_embed = nn.Embedding(w_size, w_embed_dim)
        self.t_embed = nn.Embedding(t_size, t_embed_dim)
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.linear = nn.Linear((w_embed_dim + t_embed_dim) * 10, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.target_size)

        self.w_embed.weight.data.uniform_(-0.01, 0.01)
        self.t_embed.weight.data.uniform_(-0.01, 0.01)

    def forward(self, words, tags):
        wem = self.w_embed(words).view(words.size(0), -1)
        tem = self.t_embed(tags).view(tags.size(0), -1)
        inputs = torch.cat([wem, tem], 1)
        h1 = torch.pow(self.linear(inputs), 3)
        preds = -self.out(h1)
        return F.log_softmax(preds, 1)


# Training
STEP = 5
BATCH_SIZE = 256
W_EMBED_SIZE = 50
T_EMBED_SIZE = 10
HIDDEN_SIZE = 512
LR = 0.01

model = NeuralDependencyParser(len(word2index), W_EMBED_SIZE, len(tag2index), T_EMBED_SIZE, HIDDEN_SIZE,
                               len(action2index))
model.to(device)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

losses = []
for i, batch in enumerate(get_batch(BATCH_SIZE, train_data)):
    model.zero_grad()
    inputs, targets = list(zip(*batch))
    words, tags = list(zip(*inputs))
    # print([word.shape for word in words])
    # print([tag.shape for tag in tags])
    # print([target.shape for target in targets])
    words = torch.cat(words).to(device)
    tags = torch.cat(tags).to(device)
    targets = torch.cat(targets).to(device)
    preds = model(words, tags)
    loss = loss_function(preds, targets.view(-1))
    loss.backward()
    optimizer.step()

    losses.append(loss.data.tolist())
    if i % 100 == 0:
        print('mean_loss:%.2f' % (np.mean(losses)))
        losses = []


# Test 
dev = open('./data/dev.txt', 'r').readlines()
splited_data = [[nltk.pos_tag(d.split('|||')[0].split()), d.split('|||')[1][:-1].split()] for d in dev]
dev_data = []

for tx, ty in splited_data:
    state = TransitionState(tx)
    transition = ty + ['REDUCE_R']
    while len(transition) !=0:
        features = get_feature(state, word2index, tag2index)
        action = transition.pop(0)
        dev_data.append([features, action2index[action]])
        if action == 'SHIFT':
            state.shift()
        elif action == 'REDUCE_R':
            state.right_arc()
        elif action == 'REDUCE_L':
            state.left_arc()

accuracy = 0
for dev in dev_data:
    inputs, target = dev[0], dev[1]
    word, tag = inputs[0].to(device), inputs[1].to(device)
    pred = model(word, tag).max(1)[1]
    pred = pred.data.tolist()[0]
    if pred == target:
        accuracy += 1

print(accuracy/len(dev_data) * 100)

# plotting parsed result
test = TransitionState(nltk.pos_tag("I eat an cake on my birthday".split()))
index2action = {i: v for v, i in action2index.items()}

while test.is_done() == False:
    features = get_feature(test, word2index, tag2index)
    word, tag = features[0].to(device), features[1].to(device)
    action = model(word, tag).max(1)[1].data.tolist()[0]

    action = index2action[action]
    if action == 'SHIFT':
        test.shift()
    elif action == 'REDUCE_R':
        test.right_arc()
    elif action == 'REDUCE_L':
        test.left_arc()

print(test)
test.to_tree_string()
draw_nltk_tree(Tree.fromstring(test.to_tree_string()))


# 有空整理一下重构一下，有点乱
