from text_cnn import TextCNN
import torch
import data_helpers


def test():
    accuracy = 0
    loadfile = './checkpoint/40_checkpoint.tar'
    checkpoint = torch.load(loadfile)
    model_sd = checkpoint['model']
    train_data, test_data, word2index, target2index = data_helpers.preprocess()
    print(target2index)
    model = TextCNN(len(word2index), 300, len(target2index))
    model.load_state_dict(model_sd)
    inputs, targets = data_helpers.pad_to_batch(test_data, word2index)
    pred = model(inputs).max(1)[1]
    pred = pred.data.tolist()
    target = targets.data.tolist()
    for i in range(len(pred)):
        if pred[i] == target[i]:
            accuracy += 1
    print('accuracy:{}%'.format(accuracy/len(test_data)*100))


if __name__ == '__main__':
    test()
