import random
import torch
from torch import nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, sos, eos):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.sos = sos
        self.eos = eos

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def generate_inference(self, src, src_len):
        self.eval()
        batch_size = src.shape[0]
        trg_vocab_size = self.decoder.output_dim
        trg_len = 100
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_states, hidden, cell = self.encoder(src, src_len)

        maxlen = src.shape[1]
        mask = torch.arange(maxlen)[None, :] < torch.LongTensor(src_len)[:, None]

        encoder_states = encoder_states.permute(1, 0, 2)
        x = torch.tensor([self.sos] * batch_size).to(self.device)

        for t in range(0, trg_len):
            output, hidden, cell = self.decoder(x, mask, encoder_states, hidden, cell)
            outputs[t] = output
            x = output.argmax(1)

        return outputs

    def forward(self, src, src_len, trg, teacher_force_ratio=0.0):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_states, hidden, cell = self.encoder(src, src_len)

        maxlen = src.shape[1]
        mask = torch.arange(maxlen)[None, :] < torch.LongTensor(src_len)[:, None]
        input = trg[:, 0]
        encoder_states = encoder_states.permute(1, 0, 2)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, mask, encoder_states, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)

            input = trg[:, t] if random.random() < teacher_force_ratio else top1

        return outputs
