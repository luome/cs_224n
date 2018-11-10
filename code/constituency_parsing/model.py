import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import OrderedDict


class RNTN(nn.Module):
    def __init__(self, word2index, hidden_size, output_size):
        super(RNTN, self).__init__()

        self.word2index = word2index 
        self.embed = nn.Embedding(len(word2index), hidden_size)
        self.V = nn.ParameterList([nn.Parameter(torch.randn(hidden_size*2, hidden_size*2)) for _ in range(hidden_size)])
        self.W = nn.Parameter(torch.randn(hidden_size*2, hidden_size))
        self.b = nn.Parameter(torch.randn(1, hidden_size))
        self.W_out = nn.Linear(hidden_size, output_size)
    
    def init_weight(self):
        nn.init.xavier_uniform_(self.embed.state_dict()['weight'])
        nn.init.xavier_uniform_(self.W_out.state_dict()['weight'])
        for param in self.V.parameters():
            nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.W)
        self.b.data.fill_(0)

    def tree_propagation(self, node):
        recursive_tensor = OrderedDict()
        current = None 
        if node.isLeaf:
            tensor = torch.tensor([self.word2index[node.word]], dtype=torch.long, device=torch.device('cuda')) if node.word in self.word2index.keys() else torch.tensor([self.word2index['<UNK>']], dtype=torch.long)
            current = self.embed(tensor) # 1*D
        else:
            recursive_tensor.update(self.tree_propagation(node.left))
            recursive_tensor.update(self.tree_propagation(node.right))
            concated = torch.cat([recursive_tensor[node.left], recursive_tensor[node.right]], 1) # 1*2D
            xVx = []
            for i, v in enumerate(self.V):
                xVx.append(torch.matmul(torch.matmul(concated, v), concated.transpose(0, 1)))
            
            xVx = torch.cat(xVx, 1) # 1*D
            Wx = torch.matmul(concated, self.W) # 1*D

            current = torch.tanh(xVx+Wx+self.b)
        
        recursive_tensor[node] = current
        return recursive_tensor

    def forward(self, Trees, root_only=False):
        propagated = []
        if not isinstance(Trees, list):
            Trees = [Trees]
        for Tree in Trees:
            recursive_tensor = self.tree_propagation(Tree.root)
            if root_only:
                recursive_tensor = recursive_tensor[Tree.root]
                propagated.append(recursive_tensor)
            else:
                recursive_tensor = [tensor for node, tensor in recursive_tensor.items()]
                propagated.extend(recursive_tensor)
        
        propagated = torch.cat(propagated)

        return F.log_softmax(self.W_out(propagated), 1)
        




            



