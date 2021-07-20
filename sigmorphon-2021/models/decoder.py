import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):

        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(hid_dim*2 + emb_dim, hid_dim, n_layers)
        self.energy = nn.Linear(hid_dim*3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.output_dim = output_dim

    def forward(self, input, attention_mask, encoder_states, hidden, cell):

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        energy = energy.squeeze(2).permute(1,0)
        energy[~attention_mask] = float('-inf')
        energy = energy.permute(1,0).unsqueeze(2)

        #print("energy: ", energy.squeeze(2).permute(1,0).shape)
        #print(energy.squeeze(2).permute(1, 0)[0])
        attention = self.softmax(energy)
        #print("attention: ", attention.squeeze(2).permute(1,0).shape)
        #print(attention.squeeze(2).permute(1,0)[0])
        attention = attention.permute(1, 2, 0)
        encoder_states = encoder_states.permute(1, 0, 2)
        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)

        rnn_input = torch.cat((context_vector, embedded), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        prediction = self.fc_out(output)
        prediction = prediction.squeeze(0)
        return prediction, hidden, cell
