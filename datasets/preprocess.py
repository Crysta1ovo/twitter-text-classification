import gensim
import numpy as np
from collections import Counter


def tokenize(text):
    return text.strip()


def build_vocab(corpus, min_count=5):
    words = tokenize(corpus)
    id2word = ['<pad>', '<unk>']
    id2word.extend([
        word for word, count in Counter(words).most_common()
        if count >= min_count
    ])
    vocab_size = len(id2word)
    word2id = dict(zip(id2word, range(vocab_size)))

    return id2word, word2id


def load_pretrained_embedding(path_to_file, id2word, embed_size):
    model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file)

    vocab_size = len(id2word)
    weights = np.zeros((vocab_size, embed_size))

    unk = id2word.index('<unk>')
    for i, word in enumerate(id2word[2:]):
        if word in model:
            vector = model[word]
            weights[i + 2] = vector
            weights[unk] += vector

    weights[unk] = weights[unk] / vocab_size
    weights = weights / np.std(weights)

    return weights


def load_data(path_to_file):
    tweets = list()
    labels = list()

    for line in open(path_to_file, encoding='utf8'):
        label, tweet = tokenize(line).split('\t')
        if tweet:
            tweets.append(tweet.lower())
            labels.append(int(label))

    return tweets, labels


def convert_text_to_ids(text, word2id, max_seq_len):
    words = tokenize(text)
    unk = word2id['<unk>']
    ids = [word2id.get(word, unk) for word in words]

    gap = max_seq_len - len(ids)
    if gap <= 0:
        ids = ids[:max_seq_len]
    else:
        ids = ids + [0] * gap

    return ids