import model 
import data_utils
import torch
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm 
import numpy as np 

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


HIDDEN_SIZE = 30
ROOT_ONLY = False
BATCH_SIZE = 20
EPOCH = 20
LR = 0.01
LAMBDA = 1e-5
RESCHEDULED = False

train_data = data_utils.load_trees('train')
word2index = data_utils.build_vocabulary(train_data)
model = model.RNTN(word2index, HIDDEN_SIZE, 5)
model.init_weight()
model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

flatten = lambda l:[item for sublist in l for item in sublist]
for epoch in range(EPOCH):
    losses = []

    if RESCHEDULED == False and epoch == EPOCH//2:
        LR *= 0.1
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=LAMBDA)  # L2 norm 
        RESCHEDULED = True 
    
    for i, batch in tqdm(enumerate(data_utils.get_batch(BATCH_SIZE, train_data))):
        if ROOT_ONLY:
            labels = [tree.labels[-1] for tree in batch]
            labels = torch.tensor(labels, dtype= torch.long)
        else:
            labels = [tree.labels for tree in batch]
            labels = torch.tensor(flatten(labels), dtype = torch.long)
        
        labels = labels.to(device)

        model.zero_grad()
        preds = model(batch, ROOT_ONLY)

        loss = loss_function(preds, labels)
        losses.append(loss.data.tolist())
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[%d / %d] mean_loss:%.2f' % (epoch, EPOCH, np.mean(losses)))
            loss = []

    if epoch % 5 == 0:
        torch.save({
            'model':model.state_dict(),
            'optim':optimizer.state_dict()
            }, './checkpoint/{}.checkpoint.tar'.format(epoch))          