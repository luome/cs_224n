import data_helpers
from text_cnn import TextCNN
import torch
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 


class Train(object):
    def __init__(self, model, epoch=50, batch_size=50, kernel_sizes=[3,4,5], kernel_dim=100, lr=0.001):
        self.epoch = epoch
        self.batch_size = batch_size
        self.kernel_sizes = kernel_sizes
        self.kernel_dim = kernel_dim 
        self.lr = lr
        self.model = model

    def train(self, train_data, word2index, device):
        model = self.model
        loss_funtion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        for e in range(self.epoch):
            losses = []
            for i, batch in enumerate(data_helpers.get_batch(self.batch_size, train_data)):
                inputs, targets = data_helpers.pad_to_batch(batch, word2index)
                inputs = inputs.to(device)
                targets = targets.to(device)
                model.zero_grad()
                preds = model(inputs, True)
                loss = loss_funtion(preds, targets)
                losses.append(loss.data.tolist())
                loss.backward()
                optimizer.step()
            if e % 10 == 0:
                print("[%d/%d] mean_loss: %0.2f" %(e, self.epoch, np.mean(losses)))
                losses = []
                torch.save({
                    'model':model.state_dict(),
                    'optim':optimizer.state_dict()

                },'checkpoint/{}_{}.tar'.format(e, 'checkpoint'))


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    train_data, test_data, word2index, target2index = data_helpers.preprocess()
    pretrained_vectors = data_helpers.load_word_vector(word2index)
    model = TextCNN(len(word2index), 300, len(target2index))
    model.init_weights(pretrained_vectors)
    model = model.to(device)
    train = Train(model=model)
    train.train(train_data, word2index, device=device)


if __name__ == "__main__":
    main()




