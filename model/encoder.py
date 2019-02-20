'''This code contains the encoder part of the model.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, params, highway):
        super().__init__()

        self.params = params
        self.hw1 = highway

        # encoding source and target
        self.rnns = nn.ModuleList([
            nn.LSTM(
                input_size=self.params.word_embed_size,
                hidden_size=self.params.encoder_rnn_size,
                num_layers=self.params.encoder_num_layers,
                batch_first=True,
                bidirectional=True)
        ] for i in range(2))

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 4, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 4, self.params.latent_variable_size)

    def forward(self, input_source, input_target):
        # input_source is of shape [batch_size, seq_len, embedding_size]
        # input_target is of shape [batch_size, seq_len, embedding_size]

        # initial state for encoding the source sentence is zeros
        # after encoding the source sentence, the state can be used to encode the paraphrase
        state = None
        for i, input in enumerate([input_source, input_target]):
            # apply highway and rnn to source(original sentence) and target(paraphrase sentence)
            [batch_size, seq_len, embedding_size] = input.size()

            input = input.view(-1, embedding_size)
            input = self.hw1(input)
            input = input.view(batch_size, seq_len, embedding_size)

            _, state = self.rnns[i](input, state)

        # final state after encoding the original and paraphrase
        [h_state, c_state] = state
        
        # consider only the final layer hidden state and context state
        h_state = h_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)[-1]
        c_state = c_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)[-1]

        # make the hidden state and context state batch major
        h_state = h_state.permute(1, 0, 2).contiguous().view(batch_size, -1)
        c_state = c_state.permute(1, 0, 2).contiguous().view(batch_size, -1)

        # concat the hidden and context states
        final_state = torch.cat([h_state, c_state], 1)

        # latent parameters => [batch_size, latent_size]
        mu = self.context_to_mu(final_state)
        logvar = self.context_to_logvar(final_state)

        return mu, logvar
