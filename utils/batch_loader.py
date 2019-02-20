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
        # list of 2 dataframes, 1 containing the original sentence, other containing the paraphrase sentence.
        self.data = [pd.DataFrame(), pd.DataFrame()]

        if self.datatype == 'quora':
            self.quora = self.get_quora()
            print(f'QUORA: train: {len(self.quora[0])}, test: {len(self.quora[1])}')
            self.data = [d.append(q, ignore_index=True) for d, q in zip(self.data, self.quora)]
        print(f'ALL: train: {len(self.data[0])}, test: {len(self.data[1])}')

    def get_sentences_from_data(self):
        # make the list of sentences from the data
        sentences = []
        for df in self.data:
            sentences += list(df['question1'].values) + list(df['question2'].values)
        return sentences

    def get_word_dict(self, sentences):
        # create the word dict from the sentences
        # word_dict contains all the words from the sentences
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
        # build the words vectors for words in word_dict which are in glove
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
        # build the word vectors for the top vocab_size words
        text = ' '.join(sentences).split()
        word_counts = collections.Counter(text)
        words = [x[0] for x in word_counts.most_common(self.vocab_size - 2)] + [self.unk_label] + [self.end_label]
        for word in words:
            self.word_vec[word] = np.random.randn(self.embedding_size)
        print("word vec is created using random initialization")

    def build_most_common_vocab(self, words):
        # consider the top vocab size from all the words
        # word_to_idx, idx_to word contains only top vocab_size words
        word_counts = collections.Counter(words)
        self.idx_to_word = [x[0] for x in word_counts.most_common(self.vocab_size - 2)] + [self.unk_label] + [self.end_label]
        self.word_to_idx = {self.idx_to_word[i]: i for i in range(self.vocab_size)}

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

    def get_onehot_vocab(self, ids):
        # get the one hot representations of the given list of ids
        batch_size = len(ids)
        max_seq_len = np.max([len(x) for x in ids])
        result = np.zeros((batch_size, max_seq_len, vocab_size), dtype=np.int32)
        for i in range(batch_size):
            for j in range(max_seq_len):
                if j < len(ids[i]):
                    result[i][j][ids[i][j]] = 1
                else:
                    result[i][j][self.vocab_size - 1] = 1
        return result

    def sample_word_from_distribution(self, distribution):
        # randomly sample a word from the given distribution
        assert distribution.shape[-1] == self.vocab_size
        ix = np.random.choice(range(self.vocab_size), p=distribution.ravel())
        return self.idx_to_word[ix]
    
    def likely_word_from_distribution(self, distribution):
        # get the word which has max probability from the given distribution
        assert distribution.shape[-1] == self.vocab_size
        ix = np.argmax(distribution.ravel())
        return self.idx_to_word[ix]
    
    def embed_batch(self, batch):
        # convert the given batch of sentences to its embedding format
        # input => list of sentences => [batch_size]
        # return => [batch_size, max_len, embedding_size]
        max_len = np.max([len(x) for x in batch])
        embed = np.zeros((len(batch), max_len, self.embedding_size), dtype=np.float32)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                if batch[i][j] == self.go_label or batch[i][j] == self.end_label:
                    continue
                if batch[i][j] in self.word_vec.keys():
                    embed[i, j, :] = self.word_vec[batch[i][j]]
                else:
                    embed[i, j, :] = self.word_vec['null']
        return embed

    def get_encoder_input(self, sentences):
        # add end label to each of the original, paraphrase sentence words
        # convert the words into embeddings
        return [Variable(torch.from_numpy(self.embed_batch([s + [self.end_label] for s in q]))).float() for q in sentences]

    def get_decoder_input(self, sentences):
        # add <eos> label at the end of the original sentence words
        # add <sos> label at the start of the paraphrase sentence words
        enc_input = self.embed_batch([s + [self.end_label] for s in sentences[0]])
        dec_input = self.embed_batch([[self.go_label] + s for s in sentences[1]])
        return [Variable(torch.from_numpy(enc_input)).float(), Variable(torch.from_numpy(dec_input)).float()]

    def get_target(self, sentences):
        # create the target index vectors for calculating loss
        # convert each paraphrase sentence words into its vector form

        # take the paraphrase sentenes
        sentences = sentences[1]
        max_seq_len = np.max([len(s) for s in sentences]) + 1
        target_idx = [[self.get_idx_by_word(w) for w in s] + [self.get_idx_by_word(self.end_label)] * (max_seq_len - len(s)) for s in sentences]
        return Variable(torch.from_numpy(np.array(target_idx, dtype=np.int64))).long()

    def input_from_sentences(self, sentences):
        # sentences is a list of original and paraphrase sentences => [[original_sentences], [paraphrase_sentences]]

        sentences = [[process_sentence(s).split() for s in q] for q in sentences]
        # sentences is a list of words of original and paraphrase sentences
        # sentences => [[[original_sentence_1_words], [original_sentence_2_words]], [[paraphrase_sentence_1_words], [paraphrase_sentence_2_words]]]

        encoder_input_source, encoder_input_target = self.get_encoder_input(sentences)
        decoder_input_source, decoder_input_target = self.get_decoder_input(sentences)
        target = self.get_target(sentences)

        return [
            encoder_input_source, encoder_input_target,
            decoder_input_source, decoder_input_target,
            target]

    def next_batch(self, batch_size, type, return_sentences=False, balanced=True):
        # get a batch of data from the corresponding file type => train / test
        if type == 'train':
            file_id = 0
        if type == 'test':
            file_id = 1

        if balanced:
            # sample evenly from all the available datasets
            # for now there is only quora dataset, so we will sample from only quora
            df = self.quora[file_id].sample(batch_size, replace=False)
        else:
            # read and sample from the test dataframe
            df = self.data[file_id].sample(batch_size, replace=False)
        sentences = [df['question1'].values, df['question2'].values]

        # swap source and target
        if np.random.rand() < 0.5:
            sentences = [sentences[1], sentences[0]]

        input = self.input_from_sentences(sentences)

        if return_sentences:
            return input, [[process_sentence(s).split() for s in q] for q in sentences]
        else:
            return input

if __name__ == '__main__':
    batch_loader = BatchLoader(datapath='../data/quora', glove_path='../data/glove.6B.100d.txt', use_glove=False)
    print(batch_loader.max_seq_len)
    input = batch_loader.next_batch(3, 'train')
    print('done')
