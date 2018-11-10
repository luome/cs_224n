import random 
from nltk.draw import TreeWidget 
from nltk.draw.util import CanvasFrame
from nltk.tree import Tree as nltkTree 
from tkinter import mainloop


class Node(object):
    def __init__(self, label, word=None):
        self.label = label 
        self.word = word
        self.parent = None
        self.left = None
        self.right = None  
        self.isLeaf = False

    def __str__(self):
        if self.isLeaf:
            return '[{0}:{1}]'.format(self.word, self.label)
        return '({0} <- [{1}:{2}] -> {3})'.format(self.left, self.word, self.label, self.right)


class Tree(object):
    def __init__(self, treeString, openChar='(', closeChar=')'):
        tokens = []
        self.open = '('
        self.close =')'
        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)
        self.labels = get_labels(self.root)
        self.num_words = len(self.labels)

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, 'Malformed tree'
        assert tokens[-1] == self.close, 'Malformed tree'

        split = 2 # postion after open and label
        count_open = count_close = 0
        if tokens[split] == self.open:
            count_open += 1
            split += 1
        
        # find where left child and right child split 
        while count_open != count_close:
            if tokens[split] == self.open:
                count_open += 1
            if tokens[split] == self.close:
                count_close += 1
            split += 1

        node = Node(int(tokens[1]))

        node.parent = parent

        if count_open == 0:
            node.word = ''.join(tokens[2:-1]).lower()
            node.isLeaf = True
            return node
        
        node.left = self.parse(tokens[2:split], parent=node)
        node.right = self.parse(tokens[split:-1], parent=node)

        return node 

    def get_words(self):
        leaves = get_leaves(self.root)
        words = [node.word for node in leaves]
        return words

def get_labels(node):
    if node is None:
        return []
    return get_labels(node.left) + get_labels(node.right) + [node.label]

def get_leaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return get_leaves(node.left) + get_leaves(node.right)

def load_trees(data_set='train'):
    file = './data/%s.txt' % data_set
    print('Loading %s trees...' % data_set)
    with open(file, 'r', encoding='utf-8') as fid:
        trees = [Tree(l) for l in fid.readlines()] 
    return trees 

def get_batch(batch_size, train_data, shuffle=True):
    if shuffle:
        random.shuffle(train_data)
    batch_num = int((len(train_data)-1)/batch_size) + 1
    for batch_num in range(batch_num):
        start_index = batch_size * batch_num 
        end_index = min((batch_num + 1) * batch_size, len(train_data))
        yield train_data[start_index:end_index]

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
    mainloop()

def build_vocabulary(train_data):
    flatten = lambda l:[item for sublist in l for item in sublist]
    vocab = list(flatten([t.get_words() for t in train_data]))
    word2index = {'<UNK>': 0}
    for vo in vocab:
        if word2index.get(vo) is None:
            word2index[vo] = len(word2index)
    return word2index


if __name__ == '__main__':
    sample = random.choice(open('./data/train.txt', 'r', encoding='utf-8').readlines())
    print(sample)
    draw_nltk_tree(nltkTree.fromstring(sample))
    train_data = load_trees('train')
    word2index = build_vocabulary(train_data)
    print(word2index)

    
