import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.fc_hidden = nn.Linear(hid_dim*2*self.n_layers, hid_dim)
        self.fc_cell = nn.Linear(hid_dim*2*self.n_layers, hid_dim)

    def forward(self, src, src_len):

        embedded = self.dropout(self.embedding(src))

        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len, enforce_sorted=False, batch_first=True)

        outputs, (hidden, cell) = self.rnn(packed)

        encoder_states, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, padding_value=0, batch_first=True)

        hidden = torch.cat((torch.unbind(hidden.squeeze(0))),dim=-1)
        cell = torch.cat((torch.unbind(cell.squeeze(0))),dim=-1)

        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)

        hidden = self.fc_hidden(hidden)
        cell = self.fc_cell(cell)
        #hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        #cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell
