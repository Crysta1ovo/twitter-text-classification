import torch
from torch.utils.data import Dataset


class TwitterDataset(Dataset):
    def __init__(self, tweets, labels):
        super(TwitterDataset, self).__init__()
        self.tweets = torch.LongTensor(tweets)
        self.labels = torch.Tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        label = self.labels[idx]

        return tweet, label