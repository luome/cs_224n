import torch
import torch.nn.functional as F 
import torch.nn as nn


# hyperparameters comes from the article *Convolutional Neural Networks for Sentence Classification*
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim=100, kernel_sizes=(3, 4, 5), dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # in_channel：1, out_channel: 100, kernel_size: (K, D)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes)*kernel_dim, output_size)
    
    def forward(self, inputs, is_training=False):
        inputs = self.embedding(inputs).unsqueeze(1)
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]

        concated = torch.cat(inputs, 1)
        if is_training:
            concated = self.dropout(concated)
        out = self.fc(concated)
        return F.log_softmax(out, 1)

    def init_weights(self, pretrained_word_vectors, is_static=True):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False
        
