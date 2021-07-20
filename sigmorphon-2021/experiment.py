from datetime import datetime
from torch import optim, nn
import torch
import numpy as np
import os
import yaml
import csv
import logging
import pickle

from vocabulary import Vocabulary
from data import get_data, pad_data, get_df
from batched_iterator import BatchedIterator
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq


class Experiment:
    def __init__(self, cfg):
        self.result = {"data_language": cfg.config.language}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

    def __enter__(self):
        self.result["start_time"] = datetime.now()
        self.result["running_time"] = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.result["running_time"] = (datetime.now() - self.result["start_time"]).total_seconds()

        result_file = os.path.join(os.getcwd(), "result.yaml")
        with open(result_file, 'w+') as file:
            yaml.dump(self.result, file)

    def train(self, model, X_train, y_train, X_dev, y_dev, config):
        log = logging.getLogger('train')
        optimizer = optim.Adam(model.parameters(), lr=config.lrate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        train_iter = BatchedIterator(X_train, y_train, batch_size=config.batch_size)

        dev_iter = BatchedIterator(X_dev, y_dev, batch_size=config.batch_size)

        teacher_force_ratio = 0.6

        num_epochs = config.epochs

        all_dev_loss = []
        all_dev_acc = []
        all_train_loss = []
        all_train_acc = []

        patience = config.patience
        epochs_no_improve = 0
        min_loss = np.Inf
        early_stopping = False
        best_epoch = 0

        for epoch in range(num_epochs):
            model.train()
            for bi, (batch_x, batch_y) in enumerate(train_iter.iterate_once()):
                batch_x, len_x = pad_data(batch_x, 0)
                batch_x = batch_x.to(self.device)
                batch_y, len_y = pad_data(batch_y, 0)
                batch_y = batch_y.to(self.device)
                y_out = model(batch_x, len_x, batch_y, teacher_force_ratio)
                y_out = y_out[1:].reshape(-1, y_out.shape[2])
                batch_y = batch_y.transpose(0, 1)[1:].reshape(-1)
                loss = criterion(y_out, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                if epoch % 5 == 0:
                    train_acc = 0
                    train_loss = 0
                    #train_acc, train_loss = self.evaluate_model(model, train_iter, criterion)
                else:
                    train_acc = 0
                    train_loss = 0
                dev_acc, dev_loss = self.evaluate_model(model, dev_iter, criterion)

                all_train_acc.append(train_acc)
                all_train_loss.append(train_loss)
                all_dev_loss.append(dev_loss)
                all_dev_acc.append(dev_acc)

                if epoch % 15 == 0:
                    teacher_force_ratio -= 0.05

                #print(f"Epoch: {epoch}")
                log.info(f"Epoch: {epoch}")
                #print(f"  train accuracy: {train_acc}  train loss: {train_loss}")
                log.info(f"  train accuracy: {train_acc}  train loss: {train_loss}")
                #print(f"  dev accuracy: {dev_acc}  dev loss: {dev_loss}")
                log.info(f"  dev accuracy: {dev_acc}  dev loss: {dev_loss}")
            torch.save(model, os.path.join(os.getcwd(), "model_latest.pt"))
            if min_loss - dev_loss > 0.001:
                epochs_no_improve = 0
                min_loss = dev_loss
                best_epoch = epoch
                torch.save(model, os.path.join(os.getcwd(), "model.pt"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    early_stopping = True
                    print("Early stopping")
            if early_stopping:
                break

        return all_train_acc[best_epoch], all_train_loss[best_epoch], all_dev_acc[best_epoch], \
                   all_dev_loss[best_epoch]

    def evaluate_model(self, model, iterator, criterion):
        loss = 0
        correct_guesses = 0
        all_guesses = 0
        bi = 0
        for bi, (batch_x, batch_y) in enumerate(iterator.iterate_once()):
            batch_x, len_x = pad_data(batch_x, 0)
            batch_x = batch_x.to(self.device)
            batch_y, len_y = pad_data(batch_y, 0)
            batch_y = batch_y.to(self.device)
            dev_out = model(batch_x, len_x,  batch_y, teacher_force_ratio=0)
            dev_out = dev_out[1:].reshape(-1, dev_out.shape[2])
            loss += criterion(dev_out, batch_y.transpose(0, 1)[1:].reshape(-1))

            label = batch_y.transpose(0, 1)[1:].reshape(-1)
            dev_pred = dev_out.max(axis=1)[1]
            eq = torch.eq(dev_pred, label)
            batch_mask = self.get_mask(batch_y)
            eq[batch_mask] = 0
            correct_guesses += eq.sum().float()
            all_guesses += torch.sum(batch_mask == False)
        loss /= (bi + 1)
        acc = correct_guesses / all_guesses
        return acc, loss

    def get_mask(self, tensor):
        return (tensor == 0).transpose(1, 0)[1:].reshape(-1)

    def run(self, cfg):
        train_df, dev_df = get_df(cfg.config.train, cfg.config.dev)
        
        if cfg.model.use_vocab:
            vocab_file = os.path.join(cfg.model.vocab_dir, 'vocab.pkl')
            vocab_dec_file = os.path.join(cfg.model.vocab_dir, 'vocab_dec.pkl')
            with open(vocab_file, 'rb') as file:
                vocab_enc = pickle.load(file)
            with open(vocab_dec_file, 'rb') as file:
                vocab_dec = pickle.load(file)
            vocab = Vocabulary(vocab=vocab_enc, vocab_dec=vocab_dec)
        else:
            vocab = Vocabulary(df=train_df)

        X_train, y_train, X_dev, y_dev = get_data(train_df, dev_df, vocab)

        input_size_encoder = len(vocab.char_indices)
        input_size_decoder = len(vocab.char_indices)
        encoder_embedding_size = cfg.model.enc_emb_size
        decoder_embedding_size = cfg.model.dec_emb_size
        hidden_size = cfg.model.hidden_size
        num_layers = cfg.model.num_layers
        enc_dropout = cfg.model.enc_dropout
        dec_dropout = cfg.model.dec_dropout

        if cfg.model.load_model:
            model = torch.load(cfg.model.model_file).to(self.device)
            #model.encoder = model.encoder.to(self.device)
            #model.decoder = model.decoder.to(self.device)
            #model.device = self.device
            #model.sos = vocab.char_indices['<SOS>']
            #model.eos = vocab.char_indices['<EOS>']
        else:
            encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size,
                                  num_layers, enc_dropout).to(self.device)

            decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size,
                                  num_layers, dec_dropout).to(self.device)

            model = Seq2Seq(encoder_net, decoder_net, self.device, vocab.char_indices['<SOS>'],
                            vocab.char_indices['<EOS>']).to(self.device)

        train_acc, train_loss, dev_acc, dev_loss = self.train(model, X_train, y_train, X_dev, y_dev, cfg.model)

        self.result["train_acc"] = float(train_acc)
        self.result["train_loss"] = float(train_loss)
        self.result["dev_acc"] = float(dev_acc)
        self.result["dev_loss"] = float(dev_loss)

        all_result_file = os.path.join(os.path.dirname(os.getcwd()), "all_result.csv")

        with open(all_result_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.result['data_language'], float(train_acc), float(train_loss), float(dev_acc),
                             float(dev_loss)])


