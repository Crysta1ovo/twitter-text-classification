import os
import time
import torch
import torch.nn as nn
from datasets.preprocess import *
from datasets.dataloader import TwitterDataset
from torch.utils.data import DataLoader
from models.TextCNN import TextCNN
from models.LSTM import LSTM
from models.Transformer import TransformerEncoder

gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()

if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")

print(f"Use cuda: {use_cuda}, gpu id: {gpu}.")

data_dir = 'data'

train_tweets, y_train = load_data(os.path.join(data_dir, 'train.txt'))
test_tweets, y_test = load_data(os.path.join(data_dir, 'test.txt'))

corpus = ' '.join(train_tweets)

id2word, word2id = build_vocab(corpus)

max_seq_len = 65
X_train = [
    convert_text_to_ids(tweet, word2id, max_seq_len) for tweet in train_tweets
]
X_test = [
    convert_text_to_ids(tweet, word2id, max_seq_len) for tweet in test_tweets
]

batch_size = 128
train_dataset = TwitterDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
test_dataset = TwitterDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

embed_size = 100
pretrained_embedding = load_pretrained_embedding(
    os.path.join(data_dir, 'glove-100d.txt'), id2word, embed_size)

pad = word2id['<pad>']
vocab_size = len(id2word)
dropout = 0.15
n_labels = 1

# TextCNN
# n_filters = 100
# filter_sizes = [2, 3, 4]
# model = TextCNN(pretrained_embedding, vocab_size, embed_size, n_filters,
#                 filter_sizes, n_labels, dropout, pad)

# LSTM
hidden_size = 200
n_layers = 2
bidirectional = True
model = LSTM(pretrained_embedding, vocab_size, embed_size, hidden_size,
             n_labels, n_layers, bidirectional, dropout, pad)

# Transformer
# model = TransformerEncoder(vocab_size, max_seq_len, n_labels)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(y_pred, y):
    y_pred = torch.round(torch.sigmoid(y_pred))
    correct = (y_pred == y).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def train(model, dataloader, optimizer, criterion):

    epoch_loss = 0.
    epoch_acc = 0.
    model.train()

    for (tweets, labels) in dataloader:
        print('1')
        if use_cuda:
            tweets = tweets.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        predictions = model(tweets).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def evaluate(model, dataloader, criterion):

    epoch_loss = 0.
    epoch_acc = 0.
    model.eval()

    with torch.no_grad():
        for (tweets, labels) in dataloader:

            if use_cuda:
                tweets = tweets.cuda()
                labels = labels.cuda()

            predictions = model(tweets).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


n_epochs = 10

best_valid_loss = float('inf')

for epoch in range(n_epochs):

    start_time = time.time()

    train_loss, train_acc = train(model, train_dataloader, optimizer,
                                  criterion)
    valid_loss, valid_acc = evaluate(model, val_dataloader, criterion)
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')