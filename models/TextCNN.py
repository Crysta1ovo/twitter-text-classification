import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, pretrained_embedding, vocab_size, embed_size, n_filters,
                 filter_sizes, n_labels, dropout, padding_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,
                                      embed_size,
                                      padding_idx=padding_idx)
        self.embedding.weight.data.copy_(
            torch.from_numpy(pretrained_embedding))

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(filter_size, embed_size))
            for filter_size in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text: (batch_size, sent_len)
        # embedding: (batch_size, sent_len, embed_size)
        embedding = self.dropout(self.embedding(text))
        # embedding: batch_size, 1, sent_len, embed_size)
        embedding = embedding.unsqueeze(1)
        # conved: (batch_size, n_filters, sent_len - filter_size)
        conved = [F.relu(conv(embedding)).squeeze(3) for conv in self.convs]
        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved
        ]
        # flattened: (batch size, n_filters * len(filter_sizes))
        flattened = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(flattened)
