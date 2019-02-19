import os
import re
import torch
import collections
import unicodedata
import numpy as np
import pandas as pd
from torch.autograd import Variable


def unicodeToAscii(s):
    # convert the unicode format to ascii
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def process_sentence(sentence):
    # Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(sentence.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


class BatchLoader:
    def __init__(self, vocab_size=20000, sentences=None, datatype='quora', datapath='', use_glove=True, glove_path=None, embedding_size=100):
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
            self.quora_data_files = [os.path.join(datapath, 'train.csv'), os.path.join(datapath, 'test.csv')]

        self.use_glove = use_glove
        self.glove_path = glove_path
        self.embedding_size = embedding_size

        if sentences is None:
            self.read_dataset()

        self.build_vocab(sentences)

    def get_quora(self):
        # read the quora dataset
        return [pd.read_csv(f)[['question1', 'question2']] for f in self.quora_data_files]

    def read_dataset(self):
        # read the data
        self.data = [pd.DataFrame(), pd.DataFrame()]

        if self.datatype == 'quora':
            self.data = self.get_quora()

    def get_sentences_from_data(self):
        # make the list of sentences from the data
        sentences = []
        for df in self.data:
            sentences += list(df['question1'].values) + list(df['question2'].values)
        return sentences

    def get_word_dict(self, sentences):
        # create the word dict from the sentences
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

    def build_word_vectors_with_glove(self, word_dict):
        # read the word vectors from glove
        with open(self.glove_path) as f:
            for line in f:
                words = line.split()
                word = line[0]
                if word in word_dict:
                    vector = np.asarray(words[1:], dtype=np.float32)
                    assert vector.shape[0] == self.embedding_size
                    self.word_vec[word] = vector
        print(f'Found {len(self.word_vec)/len(word_dict)} words in glove embeddings.')

    def build_word_vectors_with_random(self, sentences):
        # create word vectors using random intialization
        text = ' '.join(sentences).split()
        word_counts = collections.Counter(text)
        words = [x[0] for x in word_counts.most_common(self.vocab_size - 2)] + [self.unk_label] + [self.end_label]
        for word in words:
            self.word_vec[word] = np.random.randn(self.embedding_size)
        print("word vec is created using random initialization")

    def build_most_common_vocab(self, words):
        # consider the top vocab size from all the words
        word_counts = collections.Counter(words)
        self.idx_to_word = [x[0] for x in word_counts.most_common(self.vocab_size - 2)] + [self.unk_label] + [self.end_label]
        self.word_to_index = {self.idx_to_word[i]: i for i in range(self.vocab_size)}

    def build_input_vocab(self, sentences):
        word_dict = self.get_word_dict(sentences)
        if self.use_glove:
            self.build_word_vectors_with_glove(word_dict)
        else:
            self.build_word_vectors_with_random(sentences)
        print(f'Vocab size : {len(self.word_vec)}')

    def build_output_vocab(self, sentences):
        self.max_seq_len = np.max([len(s) for s in sentences]) + 1
        text = ' '.join(sentences).split()
        self.build_most_common_vocab(text)

    def build_vocab(self, sentences):
        # create the vocab from the sentences
        if sentences is None:
            sentences = self.get_sentences_from_data()
        sentences = [process_sentence(sentence) for sentence in sentences]

        self.build_input_vocab(sentences)
        self.build_output_vocab(sentences)

    def get_word_by_index(self, idx):
        # get the word for the given index
        return self.idx_to_word[idx]

    def get_idx_by_word(self, word):
        # get the index for the given word, if present else return unk idx
        if word in self.word_to_idx.keys():
            return self.word_to_idx[word]
        return self.word_to_idx[self.unk_label]

if __name__ == '__main__':
    batch_loader = BatchLoader(datapath='../data/quora', glove_path='../data/glove.6B.100d.txt', use_glove=False)
    print(batch_loader.max_seq_len)
