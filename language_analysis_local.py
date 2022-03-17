import numpy as np
import torch
from egg.core.callbacks import Callback, ConsoleLogger
from egg.core.interaction import Interaction
import json
import editdistance
from scipy.spatial import distance
from scipy.stats import spearmanr
from typing import Union, Callable
import pickle


class SavingConsoleLogger(ConsoleLogger):
    """Console logger that also stores the reported values"""
    def __init__(self, print_train_loss=False, as_json=False, n_metrics=2,
                 save_path: str = '', save_epoch: int = None):
        super(SavingConsoleLogger, self).__init__(print_train_loss, as_json)

        if len(save_path) > 0:
            self.save = True
            self.save_path = save_path
            self.save_epoch = save_epoch
            self.save_dict = {'loss_train': dict(),
                              'loss_test': dict()}
            for metric_idx in range(n_metrics):
                self.save_dict['metrics_train' + str(metric_idx)] = dict()
                self.save_dict['metrics_test' + str(metric_idx)] = dict()
        else:
            self.save = False

    def aggregate_print(self, loss: float, logs: Interaction, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)

        if self.as_json:
            dump.update(dict(mode=mode, epoch=epoch))
            output_message = json.dumps(dump)
        else:
            output_message = ", ".join(sorted([f"{k}={v}" for k, v in dump.items()]))
            output_message = f"{mode}: epoch {epoch}, loss {loss}, " + output_message
        print(output_message, flush=True)

        if self.save:
            self.save_dict['loss_' + mode][epoch] = loss
            for idx, metric_key in enumerate(sorted(aggregated_metrics.keys())):
                self.save_dict['metrics_' + mode + str(idx)][epoch] = aggregated_metrics[metric_key]
            if epoch == self.save_epoch:
                with open(self.save_path + '/loss_and_metrics.pkl', 'wb') as handle:
                    pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        if self.save:
            with open(self.save_path + '/loss_and_metrics.pkl', 'wb') as handle:
                pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


class MessageLengthHierarchical(Callback):
    """ For every possible number of relevant attributes, take the messages for inputs with that number of relevant
    attributes and calculate the (absolute) difference between message length and number of relevant attributes."""

    def __init__(self, n_attributes, print_train: bool = True, print_test: bool = True, is_gumbel: bool = True,
                 save_path: str = '', save_epoch: int = None):

        self.print_train = print_train
        self.print_test = print_test
        self.is_gumbel = is_gumbel
        self.n_attributes = n_attributes

        if len(save_path) > 0 and save_epoch:
            self.save = True
            self.save_path = save_path
            self.save_epoch = save_epoch
            self.save_dict = {'message_length_train': dict(),
                              'message_length_test': dict()}
        else:
            self.save = False

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if self.print_train:
            self.print_difference_length_relevance(logs, 'train', epoch)

    def on_test_end(self, loss, logs, epoch):
        self.print_difference_length_relevance(logs, 'test', epoch)

    @staticmethod
    def compute_message_length(messages):

        max_len = messages.shape[1]
        # replace all symbols with zeros from first zero element
        for m_idx, m in enumerate(messages):
            first_zero_index = torch.where(m == 0)[0][0]
            messages[m_idx, first_zero_index:] = torch.zeros((1, max_len - first_zero_index))
        # calculate message length
        message_length = max_len - torch.sum(messages == 0, dim=1)
        return message_length

    @staticmethod
    def compute_message_length_hierarchical(messages, relevance_vectors):

        message_length = MessageLengthHierarchical.compute_message_length(messages)
        n_attributes = relevance_vectors.shape[1]
        number_same = torch.sum(1 - relevance_vectors, dim=1)

        message_lengths = []
        for n in range(1, n_attributes + 1):
            hierarchical_length = message_length[number_same == n].float()
            message_lengths.append(hierarchical_length)
        message_length_step = [round(torch.mean(message_lengths[i]).item(), 3) for i in range(n_attributes)]

        return message_length_step

    def print_difference_length_relevance(self, logs: Interaction, tag: str, epoch: int):

        message = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        relevance_vector = logs.sender_input[:, -self.n_attributes:]

        message_length_step = self.compute_message_length_hierarchical(message, relevance_vector)

        output = json.dumps(dict(message_length_hierarchical=message_length_step, mode=tag, epoch=epoch))
        print(output, flush=True)

        if self.save:
            self.save_dict['message_length_' + tag][epoch] = message_length_step
            if epoch == self.save_epoch:
                with open(self.save_path + '/message_length_hierarchical.pkl', 'wb') as handle:
                    pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        if self.save:
            with open(self.save_path + '/message_length_hierarchical.pkl', 'wb') as handle:
                pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def encode_input_for_topsim_hierarchical(sender_input, dimensions):
    n_features = np.sum(dimensions)
    n_attributes = len(dimensions)
    relevance_vectors = sender_input[:, -n_attributes:]
    sender_input_encoded = torch.zeros((len(sender_input), n_features + n_attributes))

    base_count = 0
    for i, dim in enumerate(dimensions):
        sender_input_encoded[relevance_vectors[:, i] == 0, base_count + i:base_count + i + dim] = (
            sender_input[relevance_vectors[:, i] == 0, base_count:base_count + dim])
        sender_input_encoded[relevance_vectors[:, i] == 1, base_count + i + dim] = 1
        base_count = base_count + dim

    return sender_input_encoded


class TopographicSimilarityHierarchical(Callback):

    def __init__(
            self,
            dimensions,
            sender_input_distance_fn: Union[str, Callable] = "hamming",
            message_distance_fn: Union[str, Callable] = "edit",
            compute_topsim_train_set: bool = True,
            compute_topsim_test_set: bool = True,
            is_gumbel: bool = False,
            save_path: str = '',
            save_epoch: int = None,
    ):

        self.sender_input_distance_fn = sender_input_distance_fn
        self.message_distance_fn = message_distance_fn

        self.compute_topsim_train_set = compute_topsim_train_set
        self.compute_topsim_test_set = compute_topsim_test_set
        assert compute_topsim_train_set or compute_topsim_test_set

        self.is_gumbel = is_gumbel
        self.dimensions = dimensions

        if len(save_path) > 0 and save_epoch:
            self.save = True
            self.save_path = save_path
            self.save_epoch = save_epoch
            self.save_dict = {'topsim_train': dict(),
                              'topsim_test': dict()}
        else:
            self.save = False

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.compute_topsim_train_set:
            self.print_message(logs, "train", epoch)

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        if self.compute_topsim_test_set:
            self.print_message(logs, "test", epoch)

    @staticmethod
    def compute_topsim(
            meanings: torch.Tensor,
            messages: torch.Tensor,
            meaning_distance_fn: Union[str, Callable] = "hamming",
            message_distance_fn: Union[str, Callable] = "edit",
    ) -> float:

        distances = {
            "edit": lambda x, y: editdistance.eval(x, y) / ((len(x) + len(y)) / 2),
            "cosine": distance.cosine,
            "hamming": distance.hamming,
            "jaccard": distance.jaccard,
            "euclidean": distance.euclidean,
        }

        meaning_distance_fn = (
            distances.get(meaning_distance_fn, None)
            if isinstance(meaning_distance_fn, str)
            else meaning_distance_fn
        )
        message_distance_fn = (
            distances.get(message_distance_fn, None)
            if isinstance(message_distance_fn, str)
            else message_distance_fn
        )

        assert (
                meaning_distance_fn and message_distance_fn
        ), f"Cannot recognize {meaning_distance_fn} \
            or {message_distance_fn} distances"

        meaning_dist = distance.pdist(meanings, meaning_distance_fn)
        message_dist = distance.pdist(messages, message_distance_fn)

        topsim = spearmanr(meaning_dist, message_dist, nan_policy="raise").correlation

        return topsim

    def print_message(self, logs: Interaction, mode: str, epoch: int) -> None:

        messages = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        messages = messages[0:1000]
        messages = [msg.tolist() for msg in messages]
        sender_input = logs.sender_input[0:1000]

        encoded_sender_input = encode_input_for_topsim_hierarchical(sender_input, self.dimensions)
        topsim = self.compute_topsim(encoded_sender_input, messages)
        output = json.dumps(dict(topsim=topsim, mode=mode, epoch=epoch))

        print(output, flush=True)

        if self.save:
            self.save_dict['topsim_' + mode][epoch] = topsim
            if epoch == self.save_epoch:
                with open(self.save_path + '/topsim.pkl', 'wb') as handle:
                    pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        if self.save:
            with open(self.save_path + '/topsim.pkl', 'wb') as handle:
                pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)