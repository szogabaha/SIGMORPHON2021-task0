import os
import pickle


class Vocabulary:
    def __init__(self, **kwargs):
        if 'df' in kwargs:
            df = kwargs['df']
            self.chars = self.create_vocab(df)
            self.char_indices = dict((c, i+5) for i, c in enumerate(self.chars))
            self.char_indices['<PAD>'] = 0
            self.char_indices['<SOS>'] = 1
            self.char_indices['<EOS>'] = 2
            self.char_indices['<SEP>'] = 3
            self.char_indices['<UNK>'] = 4
            with open(os.getcwd() + '/vocab.pkl', 'wb') as file:
                pickle.dump(self.char_indices, file)
            self.indices_char = dict((i+5, c) for i, c in enumerate(self.chars))
            self.indices_char[0] = '<PAD>'
            self.indices_char[1] = '<SOS>'
            self.indices_char[2] = '<EOS>'
            self.indices_char[3] = '<SEP>'
            self.indices_char[4] = '<UNK>'
            with open(os.getcwd() + '/vocab_dec.pkl', 'wb') as file:
                pickle.dump(self.indices_char, file)
        elif 'vocab' in kwargs and 'vocab_dec' in kwargs:
            self.char_indices = kwargs['vocab']
            self.indices_char = kwargs['vocab_dec']


    def create_vocab(self, df):
        vocab = set()

        for token in df.lemma:
            vocab |= set(token)

        for token in df.infl:
            vocab |= set(token)

        for token in df.tags:
            for tok in token.split(';'):
                vocab.add(tok)
        return vocab

    # lemma tags -> lemma<SEP>tags
    def encode_source(self, lemma, tags):
        ids = []
        for c in lemma:
            ids.append(self.char_indices.get(c, self.char_indices['<UNK>']))

        ids.append(self.char_indices['<SEP>'])

        for c in tags:
            ids.append(self.char_indices.get(c, self.char_indices['<UNK>']))

        return ids

    # infl -> <SOS>infl<EOS>
    def encode_target(self, inflection):
        ids = []
        ids.append(self.char_indices['<SOS>'])

        for c in inflection:
            ids.append(self.char_indices.get(c, self.char_indices['<UNK>']))

        ids.append(self.char_indices['<EOS>'])

        return ids

    def decode_output(self, output):
        chars = []
        for y in output:
            if y == self.char_indices['<EOS>']:
                break
            chars.append(self.indices_char.get(y, '<UNK>'))

        return ''.join(chars)
