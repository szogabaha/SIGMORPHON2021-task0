import pandas as pd
import numpy as np
import torch


def get_df(train_path, dev_path):
    train_df = pd.read_csv(train_path, sep='\t', header=None, names=['lemma', 'infl', 'tags'])
    dev_df = pd.read_csv(dev_path, sep='\t', header=None, names=['lemma', 'infl', 'tags'])

    train_df = train_df.drop_duplicates()
    dev_df = dev_df.drop_duplicates()

    train_df = train_df.replace(np.nan, 'nan', regex=True)
    dev_df = dev_df.replace(np.nan, 'nan', regex=True)

    return train_df, dev_df


def get_data(train_df, dev_df, vocab):
    train_df['tgt_encoded'] = train_df.apply(lambda x: vocab.encode_target(x.infl), axis=1)
    train_df['src_encoded'] = train_df.apply(lambda x: vocab.encode_source(x.lemma, x.tags.split(';')), axis=1)

    dev_df['tgt_encoded'] = dev_df.apply(lambda x: vocab.encode_target(x.infl), axis=1)
    dev_df['src_encoded'] = dev_df.apply(lambda x: vocab.encode_source(x.lemma, x.tags.split(';')), axis=1)

    train_df = train_df.sample(frac=1)

    X_train = train_df.src_encoded.to_numpy()
    y_train = train_df.tgt_encoded.to_numpy()

    X_dev = dev_df.src_encoded.to_numpy()
    y_dev = dev_df.tgt_encoded.to_numpy()

    return X_train, y_train, X_dev, y_dev


def pad_data(batch, pad):
    seq_len = list(map(len, batch))
    length = max(seq_len)
    data = torch.tensor([xi + [pad] * (length - len(xi)) for xi in batch])
    return data, seq_len
