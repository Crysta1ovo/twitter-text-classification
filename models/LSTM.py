import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, pretrained_embedding, vocab_size, embed_size,
                 hidden_size, n_labels, n_layers, bidirectional, dropout,
                 padding_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,
                                      embed_size,
                                      padding_idx=padding_idx)
        self.embedding.weight.data.copy_(
            torch.from_numpy(pretrained_embedding))

        self.rnn = nn.LSTM(embed_size,
                           hidden_size,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, n_labels)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def attention_net(self, output, hidden):
        # output: (batch_size, sent_len, hidden_size * n_directions)
        # hidden: (batch_size, hidden_size * n_directions)
        attn_weights = nn.Softmax(dim=1)(torch.bmm(
            output, hidden.unsqueeze(2)))  # (batch_size, sent_len, 1)
        context_vector = torch.bmm(output.transpose(
            1, 2), attn_weights).squeeze(
                2)  # (batch_size, hidden_size * n_directions)

        return context_vector

    def forward(self, text, n_words):
        # text: (batch_size, sent_len)
        # embedding: (batch_size, sent_len, embed_size)
        embedding = self.dropout(self.embedding(text))
        # embedding: (sent_len, batch_size, embed_size)
        embedding = embedding.permute(1, 0, 2)

        packed_words = pack_padded_sequence(embedding,
                                            lengths=n_words,
                                            enforce_sorted=False)

        output, (hidden, cell) = self.rnn(packed_words)
        # output: (sent_len, batch_size, hidden_size * n_directions)
        # hidden: (n_layers * n_directions, batch_size, hidden_size)
        # cell: (n_layers * n_directions, batch_size, hidden_size)
        output, _ = pad_packed_sequence(output)
        # output: (batch_size, sent_len, hidden_size * n_directions)
        output = output.permute(1, 0, 2)
        # hidden: (batch_size, hidden_size * n_directions)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]),
                           dim=1).squeeze(0)
        context_vector = self.attention_net(output, hidden)

        return self.fc(context_vector)
