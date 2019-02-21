'''This code contains the decoder part.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, params, highway):
        super().__init__()

        self.params = params
        self.hw1 = highway

        self.encoding_rnn = nn.LSTM(
            input_size=self.params.word_embed_size,
            hidden_size=self.params.encoder_rnn_size,
            num_layers=self.params.encoder_num_layers,
            batch_first=True,
            bidirectional=True)

        self.decoding_rnn = nn.LSTM(
            input_size=self.params.latent_variable_size + self.params.word_embed_size,
            hidden_size=self.params.decoder_rnn_size,
            num_layers=self.params.decoder_num_layers,
            batch_first=True)

        self.h_to_initial_state = nn.Linear(self.params.encoder_rnn_size * 2, self.params.decoder_num_layers * self.params.decoder_rnn_size)
        self.c_to_initial_state = nn.Linear(self.params.encoder_rnn_size * 2, self.params.decoder_num_layers * self.params.decoder_rnn_size)

        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.vocab_size)

    def build_initial_state(self, input):
        # run the encoder rnn on the input to produce the hidden and cell state for paraphrase decoder
        [batch_size, seq_len, embed_size] = input.size()
        input = input.view(-1, embed_size)
        input = self.hw1(input)
        input = input.view(batch_size, seq_len, embed_size)

        _, cell_state = self.encoding_rnn(input)
        [h_state, c_state] = cell_state

        # consider only the final layer hidden and cell state
        h_state = h_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)[-1]
        c_state = c_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)[-1]

        # convert to batch major format
        h_state = h_state.permute(1, 0, 2).contiguous().view(batch_size, -1)
        c_state = c_state.permute(1, 0, 2).contiguous().view(batch_size, -1)

        # pass the hidden and cell state through a linear layer to get initial hidden and cell states for decoder
        h_initial = self.h_to_initial_state(h_state).view(batch_size, self.params.decoder_num_layers, self.params.decoder_rnn_size)
        # h_inital => [num_layers, batch_size, decoder_rnn_size]
        h_initial = h_initial.permute(1, 0, 2).contiguous()

        # pass the hidden and cell state through a linear layer to get initial hidden and cell states for decoder
        c_initial = self.h_to_initial_state(c_state).view(batch_size, self.params.decoder_num_layers, self.params.decoder_rnn_size)
        # c_inital => [num_layers, batch_size, decoder_rnn_size]
        c_initial = c_initial.permute(1, 0, 2).contiguous()

        return (h_initial, c_initial)

    def forward(self, encoder_input, decoder_input, z, drop_prob, initial_state=None):
        # encoder_input shape => [batch_size, seq_len, embed_size]
        # decoder_input shape => [batch_size, seq_len, embed_size]
        # z is latent variable shape => [batch_size, latent_variable_size]
        # drop_prob is the probability of an element of decoder input to be zeroed in sense of dropout
        # initial_state is the initial state of the decoder

        if initial_state is None:
            assert encoder_input is not None
            initial_state = self.build_initial_state(encoder_input)

        [batch_size, seq_len, _] = decoder_input.size()

        # replicate the z for seq_len
        z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)

        # concatenate the latent variable with decoder input
        decoder_input = torch.cat([decoder_input, z], 2)

        # run the decoder rnn
        rnn_out, final_state = self.decoding_rnn(decoder_input, initial_state)

        # run the output layer
        # rnn_out => [batch_size, seq_len, decoder_rnn_size]
        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)
        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.vocab_size)

        return result, final_state
