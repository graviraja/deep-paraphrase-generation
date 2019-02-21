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
    parser.add_argument('--trained-model-path', type=str, default='', metavar='TM', help='path to pretrained model (default: "")')
    parser.add_argument('--model-name', default='model', metavar='MN', help='name of the model to save (default: model)')
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
    paraphraser = Paraphraser(parameters, device).to(device)

    cross_entropy_result_train = []
    kld_result_train = []
    cross_entropy_result_valid = []
    kld_result_valid = []
    cross_entropy_cur_train = []
    kld_cur_train = []

    if args.use_trained:
        # load the pretrained model
        paraphraser.load_state_dict(torch.load(args.pretrained_model_name))
        pass

    # define the optimizer
    optimizer = Adam(paraphraser.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # get the training method of the model
    train_step = paraphraser.trainer(optimizer, batch_loader)

    # get the validation method of the model
    validate = paraphraser.validator(batch_loader)

    for iteration in range(args.num_iterations):
        # run the train step
        cross_entropy, kld, coef = train_step(iteration, args.batch_size, args.dropout)

        cross_entropy_cur_train += [cross_entropy.data.cpu().numpy()]
        kld_cur_train += [kld.data.cpu().numpy()]

        # validation
        if iteration % 500 == 0:
            cross_entropy_result_train += [np.mean(cross_entropy_cur_train)]
            kld_result_train += [np.mean(kld_cur_train)]
            cross_entropy_cur_train = []
            kld_cur_train = []

            print('\n')
            print('------------------------------------------- TRAIN -------------------------------------------------------------------------')
            print(f"ITERATION = {iteration}, CROSS-ENTROPY = {cross_entropy_result_train[-1]}, KLD = {kld_result_train[-1]}, KLD-COEF = {coef}")
            print('------------------------------------------- VALID --------------------------------------------------------------------------')

            # run the validation for several batches
            cross_entropy, kld = [], []
            for i in range(5):
                ce, kl, _ = validate(args.batch_size)
                cross_entropy += [ce.data.cpu().numpy()]
                kld += [kl.data.cpu().numpy()]
            
            cross_entropy = np.mean(cross_entropy)
            kld = np.mean(kld)
            cross_entropy_result_valid += [cross_entropy]
            kld_result_valid += [kld]

            print(f"CROSS-ENTROPY = {cross_entropy}, KLD = {kld}")
            print('---------------------------------------------------------------------------------------------------------------------------')

            _, _, (sampled, s1, s2) = validate(2, need_samples=True)
            for i in range(len(sampled)):
                result = paraphraser.sample_with_pair(batch_loader, 20, s1[i], s2[i])
                print(f"source: {s1[i]}")
                print(f"target: {s2[i]}")
                print(f"valid: {sampled[i]}")
                print(f"sampled: {result}")
                print('--------------------------------------------------------------------------------------------------')

        # save the model for 10K iterations
        if (iteration % 10000 == 0 and iteration != 0) or iteration == (args.num_iterations - 1):
            torch.save(paraphraser.state_dict(), 'saved_models/trained_paraphraser_' + str(iteration) + args.model_name)
            # save the logs of cross entropy and kld loss as well
            np.save(f'logs/cross_entropy_result_valid_{iteration}_{args.model_name}.npy', np.array(cross_entropy_result_valid))
            np.save(f'logs/kld_result_valid_{iteration}_{args.model_name}.npy', np.array(kld_result_valid))
            np.save(f'logs/cross_entropy_result_train_{iteration}_{args.model_name}.npy', np.array(cross_entropy_result_train))
            np.save(f'logs/kld_result_train_{iteration}_{args.model_name}.npy', np.array(kld_result_train))
