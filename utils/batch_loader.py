import os
import re
import torch
import collections
import unicodedata
import numpy as np
import pandas as pd
from torch.autograd import Variable


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def process_str(string):
    # Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


class BatchLoader:
    def __init__(self, vocab_size=20000, sentences=None, datatype='quora', datapath='', glove_path=None):
        '''Build vocab for sentences, if sentences are none build vocab from datapath.

        '''
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_vec = {}
        self.max_seq_len = 0

        self.unk_label = '<unk>'
        self.end_label = '<eos>'
        self.go_label = '<sos>'
        self.datatype = datatype
        if self.datatype == 'quora':
            self.quora_data_files = [datapath + 'train.csv', datapath + 'test.csv']
        self.glove_path = glove_path

        if sentences is None:
            self.read_dataset(datapath)
        
        self.build_vocab(sentences)
    
    def get_quora(self):
        return [pd.read_csv(f)[['question1', 'question2']] for f in self.quora_data_files]

    def read_dataset(self):
        self.data = [pd.DataFrame(), pd.DataFrame()]

        if self.datatype == 'quora':
            self.quora = self.get_quora()

    def get_sentences_from_data(self):
        sentences = []
        for df in self.data:
            sentences += list(df['question1'].values) + list(df['question2'].values)
        return sentences

    def get_word_dict(self, sentences):
        word_dict = {}
        for sent in sentences:
            for word in sent.split():
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<sos>'] = ''
        word_dict['<eos>'] = ''
        word_dict['<unk>'] = ''
        word_dict['null'] = ''
        return word_dict

    def build_glove(self, word_dict):
        with open(self.glove_path) as f:
            for line in f:
                words = line.split()
                word = line[0]
                if word in word_dict:
                    self.word_vec[word] = np.asarray(words[1:], dtype=np.float32)
        print(f'Found {len(self.word_vec)/len(word_dict)} words in glove embeddings.')

    def build_most_common_vocab(self, words):
        word_counts = collections.Counter(words)
        self.idx_to_word = [x[0] for x in word_counts.most_common(self.vocab_size - 2)] + [self.unk_label] + [self.eos_label]
        self.word_to_index = {self.idx_to_word[i]: i for i in range(self.vocab_size)}

    def build_input_vocab(self, sentences):
        word_dict = self.get_word_dict(sentences)
        self.build_glove(word_dict)
        print(f'Vocab size : {len(self.word_vec)}')

    def build_output_vocab(self, sentences):
        self.max_seq_len = np.max([len(s) for s in sentences]) + 1
        text = ' '.join(sentences).split()
        self.build_most_common_vocab(text)

    def build_vocab(self, sentences):
        if sentences is None:
            sentences = self.get_sentences_from_data()
        sentences = [process_str(sentence) for sentence in sentences]

        self.build_input_vocab(sentences)
        self.build_output_vocab(sentences)

