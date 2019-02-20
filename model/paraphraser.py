'''This code contains the main module of the paraphrase generator.

'''
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .highway import Highway
from .encoder import Encoder
from .decoder import Decoder


class Paraphraser(nn.Module):
    def __init__(self, params, device):
        super().__init__()

        self.params = params
        self.device = device

        # use 2 layers, relu activation for Highway
        self.highway = Highway(self.params.word_embed_size, 2, F.relu)
        self.encoder = Encoder(self.params, self.highway)
        self.decoder = Decoder(self.params, self.highway)

    def forward(self, drop_prob, encoder_input=None, decoder_input=None, z=None, initial_state=None):
        # encoder_input: A list of 2 tensors (original sentence, paraphrase sentence) with shape => [batch_size, seq_len]
        # decoder_input: A list of 2 tensors (original sentence, paraphrase sentence) with shape => [batch_size, max_seq_len + 1]
        # initial_state: initial state of decoder rnn in order to perform sampling
        # drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        # z: context if sampling is performing

        if z is None:
            # training case, get context from encoder and sample z ~ N(mu, std)
            [batch_size, _, _] = encoder_input[0].size()

            mu, log_var = self.encoder(encoder_input[0], encoder_input[1])
            std = torch.exp(0.5 * log_var)

            z = Variable(torch.randn([batch_size, self.params.latent_variable_size])).to(self.device)
            z = z * std + mu

            # kl divergence loss
            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
        else:
            kld = None

        out, final_state = self.decoder(decoder_input[0], decoder_input[1], z, drop_prob, initial_state)

        return out, final_state, kld

    def learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):
        # optimizer to optimize the model
        # batch_loader, which contains all the data related things

        # train method of the model
        def train(i, batch_size, dropout):
            # i is the iteration count
            # batch_size for the iteration
            # dropout to use in the decoder
            input = batch_loader.next_batch(batch_size, 'train')
            input = [var.to(self.device) for var in input]

            [encoder_input_source, encoder_input_target, decoder_input_source, decoder_input_target, target] = input

            logits, _, kld = self(dropout, (encoder_input_source, encoder_input_target), (decoder_input_source, decoder_input_target), z=None)

            logits = logits.view(-1, self.params.vocab_size)
            target = target.view(-1)

            cross_entropy_loss = F.cross_entropy(logits, target)

            # total loss is weighted reconstruction loss + weighted kl divergence loss
            loss = self.params.cross_entropy_penalty_weight * cross_entropy_loss + self.params.get_kld_coef(i) * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return cross_entropy_loss, kld, self.params.get_kld_coef(i)
        return train

    def validator(self, batch_loader):
        # we don't need to optimize the model during validation so optimizer not required
        # batch_loader, which contains all the data related things

        def get_samples(logits, target):
            # get the sample from the logits
            # logits => [batch_size, seq_len, vocab_size]
            # target => [batch_size, seq_len]

            prediction = F.softmax(logits, dim=-1).data.cpu().numpy()
            target = target.data.cpu().numpy()

            sampled, expected = [], []
            for i in range(prediction.shape[0]):
                sampled += [' '.join([batch_loader.sample_word_from_distribution(d) for d in prediction[i]])]
                expected += [' '.join([batch_loader.get_word_by_idx(idx) for idx in target[i]])]
            return sampled, expected

        # validation loop of the model
        def validate(batch_size, need_samples=False):
            # batch_size for the validation
            # need_samples, where to have the sentences converted or not
            if need_samples:
                input, sentences = batch_loader.next_batch(batch_size, 'test', return_sentences=True)
                sentences = [[' '.join(s) for s in q] for q in sentences]
            else:
                input = batch_loader.next_batch(batch_size, 'test')

            input = [var.to(self.device) for var in input]

            [encoder_input_source, encoder_input_target, decoder_input_source, decoder_input_target, target] = input

            # consider all the words during validation
            logits, _, kld = self(0., (encoder_input_source, encoder_input_target), (decoder_input_source, decoder_input_target), z=None)

            if need_samples:
                [s1, s2] = sentences
                sampled, _ = get_samples(logits, target)
            else:
                s1, s2 = (None, None)
                sampled, _ = (None, None)

            logits = logits.view(-1, self.params.vocab_size)
            target = target.view(-1)

            cross_entropy_loss = F.cross_entropy(logits, target)

            return cross_entropy_loss, kld, (sampled, s1, s2)
        return validate

    def sample_with_input(self, batch_loader, seq_len, use_mean, input):
        [encoder_input_source, encoder_input_target, decoder_input_source, _, _] = input

        encoder_input = [encoder_input_source, encoder_input_target]

        [batch_size, _, _] = encoder_input[0].size()

        mu, logvar = self.encoder(encoder_input[0], encoder_input[1])
        std = torch.exp(0.5 * logvar)

        if use_mean:
            z = mu
        else:
            z = Variable(torch.randn([batch_size, self.params.lantent_variable_size])).to(self.device)
            z = z * std + mu

        initial_state = self.decoder.build_initial_state(decoder_input_source)

        # initial input token to the decoder is the go label
        decoder_input = batch_loader.get_raw_input_from_sentences([batch_loader.go_label])

        result = ''
        # run for seq len steps
        for i in range(seq_len):
            decoder_input = decoder_input.to(self.device)

            logits, initial_state = self.decoder(None, decoder_input, z, 0.0, initial_state)
            logits = logits.view(-1, self.params.vocab_size)

            prediction = F.softmax(logits, dim=-1)
            word = batch_loader.likely_word_from_distribution(prediction.data.cpu().numpy()[-1])

            if word == batch_loader.end_label:
                break
            result += ' ' + word

            decoder_input = batch_loader.get_raw_input_from_sentences([word])
        return result

    def sample_with_pair(self, batch_loader, seq_len, source_sent, target_sent):
        input = batch_loader.input_from_sentences([[source_sent], [target_sent]]).to(self.device)

        return self.sample_with_input(batch_loader, seq_len, False, input)
