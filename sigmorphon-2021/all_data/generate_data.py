import argparse
import os
import random

import pandas as pd
import numpy as np

lang_families = {'arz': 'Afro-Asiatic', 'afb': 'Afro-Asiatic', 'ara': 'Afro-Asiatic', 'syc': 'Afro-Asiatic',
                     'heb': 'Afro-Asiatic', 'amh': 'Afro-Asiatic',
                     'gup': 'Arnhem',
                     'aym': 'Aymaran', 'cni': 'Arawakan', 'ame': 'Arawakan',
                     'see': 'Iroquoian',
                     'sah': 'Turkic', 'tyv': 'Turkic',
                     'itl': 'Chukotko-Kamchatkan', 'ckt': 'Chukotko-Kamchatkan',
                     'evn': 'Tungusic',
                     'ckb': 'Indo-European', 'kmr': 'Indo-European', 'pol': 'Indo-European', 'rus': 'Indo-European',
                     'ces': 'Indo-European', 'bul': 'Indo-European', 'deu': 'Indo-European', 'nld': 'Indo-European',
                     'spa': 'Indo-European', 'por': 'Indo-European', 'bra': 'Indo-European', 'mag': 'Indo-European',
                     'ind': 'Austronesian', 'kod': 'Austronesian',
                     'ail': 'Trans–New Guinea',
                     'vep': 'Uralic', 'krl': 'Uralic', 'lud': 'Uralic', 'olo': 'Uralic'}


def add_lang_tag(tags, lang):
    if lang != 'all':
        family = lang_families[lang]
        tags = f"{family};{lang};{tags}"
    return tags


def gen_all_data(args):
    languages = set(file.split('.')[0] for file in os.listdir(args.src_dir))

    train_dfs = []
    dev_dfs = []

    for lang in languages:
        train_file = os.path.join(args.src_dir, f"{lang}.hall")
        dev_file = os.path.join(args.src_dir, f"{lang}.dev")
        train = pd.read_csv(train_file, sep='\t', header=None, names=['lemma', 'infl', 'tags'])
        dev = pd.read_csv(dev_file, sep='\t', header=None, names=['lemma', 'infl', 'tags'])

        train['tags'] = train.apply(lambda x: add_lang_tag(x.tags, lang), axis=1)
        dev['tags'] = dev.apply(lambda x: add_lang_tag(x.tags, lang), axis=1)

        train = train.replace(np.nan, 'nan', regex=True)
        dev = dev.replace(np.nan, 'nan', regex=True)

        train_dfs.append(train)
        dev_dfs.append(dev)

    train_df = pd.concat(train_dfs)
    dev_df = pd.concat(dev_dfs)

    trg_train_file = os.path.join(args.trg_dir, 'all.train')
    trg_dev_file = os.path.join(args.trg_dir, 'all.dev')

    train_df.to_csv(trg_train_file, sep='\t', header=False, index=False)
    dev_df.to_csv(trg_dev_file, sep='\t', header=False, index=False)


def gen_copy_data(df):
    df_lemmas = df.drop_duplicates(subset=['lemma'])
    df_lemmas['tags'] = df_lemmas.apply(lambda x: 'COPY', axis=1)
    df_lemmas['infl'] = df_lemmas.apply(lambda x: x.lemma, axis=1)
    df = pd.concat([df, df_lemmas])
    return df


def gen_double_data(df):
    df_lemmas = df.drop_duplicates(subset=['lemma'])
    df_lemmas['tags'] = df_lemmas.apply(lambda x: 'DOUBLE', axis=1)
    df_lemmas['infl'] = df_lemmas.apply(lambda x: str(x.lemma)+str(x.lemma), axis=1)
    df = pd.concat([df, df_lemmas])
    return df


def gen_tags_data(df):
    df['lemma'] = df.apply(lambda x: x.infl, axis=1)
    df['tags'] = df.apply(lambda x: 'COPY;' + x.tags)
    return df


def change_first(lemma, infl, vocab):
    letter = vocab[random.randint(0, len(vocab) - 1)]
    lemma = letter + lemma[1:]
    infl = letter + infl[1:]
    return lemma, infl


def gen_first_letters(df):
    chars = set()
    for token in df.lemma:
        chars |= set(token)

    vocab = dict((i, c) for i, c in enumerate(chars))
    first_letters = df[df.apply(lambda x: x['lemma'][0] == x['infl'][0], axis=1)]
    first_letters[['lemma', 'infl']] = first_letters.apply(lambda x: change_first(x.lemma, x.infl, vocab),
                                                           axis=1, result_type="expand")
    df = pd.concat([df, first_letters])
    return df


def get_df(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['lemma', 'infl', 'tags'])
    df = df.replace(np.nan, 'nan', regex=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir")
    parser.add_argument("--trg_dir")
    parser.add_argument("--copy", type=bool, default=False)
    parser.add_argument("--lang")
    parser.add_argument("--double_data", type=bool, default=False)
    parser.add_argument("--first_letter", type=bool, default=False)
    parser.add_argument("--gen_all", type=bool, default=False)
    args = parser.parse_args()
    if args.gen_all:
        gen_all_data(args)
    else:
        path = os.path.join(args.src_dir, f'{args.lang}.train')
        df = get_df(path)
        if args.copy:
            df = gen_copy_data(df)
        if args.double_data:
            df = gen_double_data(df)
        if args.first_letter:
            df = gen_first_letters(df)
        if args.tags:
            df = gen_tags_data(df)
        target_file = os.path.join(args.trg_dir, f'{args.lang}.train')
        df.to_csv(target_file, sep='\t', header=False, index=False)


if __name__ == '__main__':
    main()
