# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import argparse
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from language_analysis_local import *
import os
import pickle
from egg.core.language_analysis import TopographicSimilarity
from archs import Sender, Receiver
import dataset

SPLIT = (0.6, 0.2, 0.2)  # split for train, val, and test
SPLIT_ZERO_SHOT = (0.75, 0.25)  # split for train and val, test set size results from number of dimensions


def get_params(params):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--load_dataset', type=str, default=None,
                        help='If provided that data set is loaded. Data sets can be generated with pickle.ds'
                             'This makes sense if running several runs with the exact same data set.')
    parser.add_argument('--dimensions', nargs='+', type=int)
    parser.add_argument('--sample_scaling', type=int, default=10)
    parser.add_argument('--distractors', type=int, default=10)
    parser.add_argument('--vocab_size_factor', type=int, default=3,
                        help='Factor applied to minimum vocab size to calculate actual vocab size')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Size of the hidden layer of Sender and Receiver,\
                             the embedding will be half the size of hidden ')
    parser.add_argument('--sender_cell', type=str, default='gru',
                        help='Type of the cell used for Sender {rnn, gru, lstm}')
    parser.add_argument('--receiver_cell', type=str, default='gru',
                        help='Type of the cell used for Receiver {rnn, gru, lstm}')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help="Learning rate for Sender's and Receiver's parameters ")
    parser.add_argument('--temperature', type=float, default=1.5,
                        help="Starting GS temperature for the sender")
    parser.add_argument('--length_cost', type=float, default=0.0,
                        help="linear cost term per message length")
    parser.add_argument('--temp_update', type=float, default=0.99,
                        help="Minimum is 0.5")
    parser.add_argument('--save', type=bool, default=False, help="If set results are saved")
    parser.add_argument('--num_of_runs', type=int, default=5, help="How often this run should be repeated")
    parser.add_argument('--zero_shot', type=bool, default=False,
                        help="If set then zero_shot dataset will be trained and tested")
    parser.add_argument('--balanced_distractors', type=bool, default=False,
                        help="Whether distractors are sampled from all conceptual levels")

    args = core.init(parser, params)

    return args


# called for gumbel softmax
def loss(_sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    acc = (receiver_output.argmax(dim=1) == _labels).detach().float()
    loss = F.cross_entropy(receiver_output, _labels, reduction="none")
    return loss, {'acc': acc}


def train(opts, datasets, verbose_callbacks=True):

    if opts.save:
        # make folder for new run
        latest_run = len(os.listdir(opts.save_path))
        opts.save_path = os.path.join(opts.save_path, str(latest_run))
        os.makedirs(opts.save_path)
        pickle.dump(opts, open(opts.save_path + '/params.pkl', 'wb'))
        save_epoch = opts.n_epochs
    else:
        save_epoch = None

    train, val, test = datasets
    dimensions = train.dimensions

    train = torch.utils.data.DataLoader(train, batch_size=opts.batch_size, shuffle=True)
    val = torch.utils.data.DataLoader(val, batch_size=opts.batch_size, shuffle=False)
    test = torch.utils.data.DataLoader(test, batch_size=opts.batch_size, shuffle=False)

    # initialize sender and receiver agents
    sender = Sender(opts.hidden_size, sum(dimensions), len(dimensions))
    receiver = Receiver(opts.hidden_size, sum(dimensions))

    minimum_vocab_size = dimensions[0] + 1  # plus one for 'any'
    vocab_size = minimum_vocab_size * opts.vocab_size_factor + 1  # multiply by factor plus add one for eos-symbol

    # initialize game
    sender = core.RnnSenderGS(sender,
                              vocab_size,
                              int(opts.hidden_size / 2),
                              opts.hidden_size,
                              cell=opts.sender_cell,
                              max_len=len(dimensions),
                              temperature=opts.temperature)

    receiver = core.RnnReceiverGS(receiver,
                                  vocab_size,
                                  int(opts.hidden_size / 2),
                                  opts.hidden_size,
                                  cell=opts.receiver_cell)

    game = core.SenderReceiverRnnGS(sender, receiver, loss, length_cost=opts.length_cost)

    # set learning rates
    optimizer = torch.optim.Adam([
        {'params': game.sender.parameters(), 'lr': opts.learning_rate},
        {'params': game.receiver.parameters(), 'lr': opts.learning_rate}
    ])

    # setup training and callbacks
    # results/ data set name/ kind_of_dataset/ run/
    callbacks = [SavingConsoleLogger(print_train_loss=True, as_json=True,
                                     save_path=opts.save_path, save_epoch=save_epoch),
                 core.TemperatureUpdater(agent=sender, decay=opts.temp_update, minimum=0.5)]
    if opts.save:
        callbacks.extend([core.callbacks.InteractionSaver([opts.n_epochs],
                                                          test_epochs=[opts.n_epochs],
                                                          checkpoint_dir=opts.save_path),
                          core.callbacks.CheckpointSaver(opts.save_path, checkpoint_freq=0)])
    if verbose_callbacks:
        callbacks.extend([
            TopographicSimilarityHierarchical(dimensions, is_gumbel=True,
                                              save_path=opts.save_path, save_epoch=save_epoch),
            MessageLengthHierarchical(len(dimensions),
                                      print_train=True, print_test=True, is_gumbel=True,
                                      save_path=opts.save_path, save_epoch=save_epoch)
        ])

    trainer = core.Trainer(game=game, optimizer=optimizer,
                           train_data=train, validation_data=val, callbacks=callbacks)
    trainer.train(n_epochs=opts.n_epochs)

    # after training evaluate performance on the test data set
    if len(test):
        trainer.validation_data = test
        eval_loss, interaction = trainer.eval()
        acc = torch.mean(interaction.aux['acc']).item()
        print("test accuracy: " + str(acc))
        if opts.save:
            loss_and_metrics = pickle.load(open(opts.save_path + '/loss_and_metrics.pkl', 'rb'))
            loss_and_metrics['final_test_loss'] = eval_loss
            loss_and_metrics['final_test_acc'] = acc
            pickle.dump(loss_and_metrics, open(opts.save_path + '/loss_and_metrics.pkl', 'wb'))

    if not opts.zero_shot:
        # evaluate accuracy and topsim where all attributes are relevant
        max_same_indices = torch.where(torch.sum(interaction.sender_input[:, -len(dimensions):], axis=1) == 0)[0]
        acc = torch.mean(interaction.aux['acc'][max_same_indices]).item()
        sender_input = interaction.sender_input[max_same_indices]
        messages = interaction.message[max_same_indices]
        messages = messages.argmax(dim=-1)
        messages = [msg.tolist() for msg in messages]
        sender_input_hierarchical = encode_input_for_topsim_hierarchical(sender_input, dimensions)
        topsim_hierarchical = TopographicSimilarity.compute_topsim(sender_input_hierarchical,
                                                                   messages,
                                                                   meaning_distance_fn="cosine",
                                                                   message_distance_fn="edit")
        max_nsame_dict = dict()
        max_nsame_dict['acc'] = acc
        max_nsame_dict['topsim_hierarchical'] = topsim_hierarchical
        print("maximal #same eval", max_nsame_dict)

        if opts.save:
            pickle.dump(max_nsame_dict, open(opts.save_path + '/max_nsame_eval.pkl', 'wb'))


def main(params):
    opts = get_params(params)

    # has to be executed in Project directory for consistency
    assert os.path.split(os.getcwd())[-1] == 'hierarchical_reference_game'

    data_set_name = '(' + str(len(opts.dimensions)) + ',' + str(opts.dimensions[0]) + ')'
    folder_name = (data_set_name + '_sample_scaling_' + str(opts.sample_scaling) + '_balanced_' +
                   str(opts.balanced_distractors) + '_vsf_' + str(opts.vocab_size_factor))
    folder_name = os.path.join("results", folder_name)

    # if name of precreated data set is given, load dataset
    if opts.load_dataset:
        item_set = torch.load('data/' + opts.load_dataset)
        print('data loaded from: ' + 'data/' + opts.load_dataset)

    for _ in range(opts.num_of_runs):

        # otherwise generate data set
        if not opts.load_dataset:
            item_set = dataset.AnyItemSet(opts.dimensions,
                                          distractors=opts.distractors,
                                          sample_scaling=opts.sample_scaling,
                                          upsample=True,
                                          balanced_distractors=opts.balanced_distractors)
        if opts.zero_shot:
            # create subfolder if necessary
            opts.save_path = os.path.join(folder_name, 'zero_shot')
            if not os.path.exists(opts.save_path) and opts.save:
                os.makedirs(opts.save_path)
            train(opts, item_set.get_zero_shot_datasets(SPLIT_ZERO_SHOT), verbose_callbacks=False)

        else:
            # create subfolder if necessary
            opts.save_path = os.path.join(folder_name, 'standard')
            if not os.path.exists(opts.save_path) and opts.save:
                os.makedirs(opts.save_path)
            train(opts, item_set.get_datasets(SPLIT), verbose_callbacks=True)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
