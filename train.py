'''This is the starting point of the project.

'''
import argparse
import os
import numpy as np
import torch
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from model.parameters import Parameters
from model.paraphraser import Paraphraser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paraphraser')

    parser.add_argument('--num-iterations', type=int, default=60000, metavar='NI', help='num iterations (default: 60000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR', help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR', help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT', help='load pretrained model (default: False)')
    parser.add_argument('--model-name', default='paraphrase-model.pt', metavar='MN', help='name of the model to save (default: paraphrase-model.pt)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='WD', help='L2 regularization penalty (default: 0.0)')
    parser.add_argument('--data-path', type=str, default='./data/quora', metavar='DP', help='quora dataset path (default: ./data/quora)')
    parser.add_argument('--use-glove', type=bool, default=False, metavar='GV', help='use glove embeddings (default: False)')
    parser.add_argument('--glove-path', type=str, default='./data', metavar='GP', help='glove file path (default: ./data)')
    parser.add_argument('--embedding-size', type=int, default=100, metavar='ES', help='embeddings size (default: 100)')
    parser.add_argument('--interm-sampling', type=bool, default=False, metavar='IS', help='if sample while training (default: False)')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_loader = BatchLoader(datapath=args.data_path, use_glove=args.use_glove, glove_path=args.glove_path, embedding_size=args.embedding_size)
    parameters = Parameters(batch_loader.max_seq_len, batch_loader.vocab_size, batch_loader.embedding_size)
    paraphraser = Paraphraser(parameters, device)
