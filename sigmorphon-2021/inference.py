from vocabulary import Vocabulary
import numpy as np
import torch
import hydra
import pickle
from omegaconf import DictConfig
import pandas as pd
import os

from batched_iterator import BatchedIterator
from data import pad_data
from all_data.generate_data import lang_families, add_lang_tag

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_inference(model, lemma, tags, vocab, device):
    model.eval()
    input = vocab.encode_source(lemma, tags.split(';'))
    input = torch.from_numpy(np.array(input)).to(device)
    input = input.unsqueeze(0)
    length = torch.tensor([input.shape[1]]).to('cpu')
    res = model.generate_inference(input, length).transpose(0, 1).argmax(axis=2)
    for i in res[0]:
        print(vocab.indices_char[i.item()], end="")
    print()


def get_processed_data(test_df, vocab, lang):
    test_df = test_df.drop_duplicates()

    test_df = test_df.replace(np.nan, 'nan', regex=True)

    test_df['tags'] = test_df.apply(lambda x: add_lang_tag(x.tags, lang), axis=1)
    test_df['tgt_encoded'] = test_df.apply(lambda x: vocab.encode_target(x.infl), axis=1)
    test_df['src_encoded'] = test_df.apply(lambda x: vocab.encode_source(x.lemma, x.tags.split(';')), axis=1)

    X_test = test_df.src_encoded.to_numpy()

    return X_test


@hydra.main(config_name="inference_config")
def main(cfg: DictConfig):
    model_file = os.path.join(cfg.exp_dir, 'model.pt')
    model = torch.load(model_file).to(device)
    vocab_file = os.path.join(cfg.exp_dir, 'vocab.pkl')
    vocab_dec_file = os.path.join(cfg.exp_dir, 'vocab_dec.pkl')
    with open(vocab_file, 'rb') as file:
        vocab_enc = pickle.load(file)
    with open(vocab_dec_file, 'rb') as file:
        vocab_dec = pickle.load(file)

    vocab = Vocabulary(vocab=vocab_enc, vocab_dec=vocab_dec)

    test_file = os.path.join(cfg.src_dir, f'{cfg.lang}.{cfg.extension}')

    df = pd.read_csv(test_file, sep='\t', header=None)

    if len(df.columns) == 3:
        df.columns = ['lemma', 'infl', 'tags']
    else:
        df.columns = ['lemma', 'tags']

    source = get_processed_data(df, vocab, cfg.lang)
    predicted = []
    test_iter = BatchedIterator(source, batch_size=128)

    for bi, src in enumerate(test_iter.iterate_once()):
        src_padded, src_len = pad_data(src[0], vocab_enc['<PAD>'])
        src_padded = src_padded.to(device)
        outputs = model.generate_inference(src_padded, src_len)
        outputs_pred = outputs.transpose(1, 0).argmax(-1)
        for output in outputs_pred:
            predicted.append(vocab.decode_output(output.tolist()))

    pred_file = os.path.join(cfg.exp_dir, f'inference/{cfg.lang}_predicted.txt')

    os.makedirs(os.path.dirname(pred_file), exist_ok=True)

    with open(pred_file, 'w') as file:
        file.write('\n'.join(predicted))


if __name__ == '__main__':
    main()
