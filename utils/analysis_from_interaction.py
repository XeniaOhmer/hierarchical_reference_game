from egg.core.language_analysis import calc_entropy, _hashable_tensor
from sklearn.metrics import normalized_mutual_info_score
from language_analysis_local import MessageLengthHierarchical
import numpy as np
import torch


def k_hot_to_attributes(khots, dimsize):
    base_count = 0
    n_attributes = khots.shape[1] // dimsize
    attributes = np.zeros((len(khots), n_attributes))
    for att in range(n_attributes):
        attributes[:, att] = np.argmax(khots[:, base_count:base_count + dimsize], axis=1)
        base_count = base_count + dimsize
    return attributes


def joint_entropy(xs, ys):
    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    return calc_entropy(xys)


def information_scores(interaction, n_dims, n_values, normalizer="arithmetic"):
    """calculate entropy scores: mutual information (MI), effectiveness and consistency. 
    
    :param interaction: interaction (EGG class)
    :param n_dims: number of input dimensions, e.g. D(3,4) --> 3 dimensions
    :param n_values: size of each dimension, e.g. D(3,4) --> 4 values
    :param normalizer: normalizer can be either "arithmetic" -H(M) + H(C)- or "joint" -H(M,C)-
    :return: NMI, NMI per level, effectiveness, effectiveness per level, consistency, consistency per level
    """

    # Get relevant attributes
    sender_input = interaction.sender_input
    objects = sender_input[:, :-n_dims]
    intentions = sender_input[:, -n_dims:]
    messages = interaction.message.argmax(dim=-1)
    objects = k_hot_to_attributes(objects, n_values)
    objects = objects + 1
    concepts = torch.from_numpy(objects * (1 - np.array(intentions)))

    n_relevant_idx = [np.where(np.sum(1 - np.array(intentions), axis=1) == i)[0] for i in range(1, n_dims + 1)]

    m_entropy = calc_entropy(messages)
    m_entropy_hierarchical = [calc_entropy(messages[n_relevant]) for n_relevant in n_relevant_idx]
    c_entropy = calc_entropy(concepts)
    c_entropy_hierarchical = [calc_entropy(concepts[n_relevant]) for n_relevant in n_relevant_idx]
    joint_mc_entropy = joint_entropy(messages, concepts)
    joint_entropy_hierarchical = [joint_entropy(messages[n_relevant], concepts[n_relevant])
                                  for n_relevant in n_relevant_idx]

    joint_entropy_hierarchical = np.array(joint_entropy_hierarchical)
    c_entropy_hierarchical = np.array(c_entropy_hierarchical)
    m_entropy_hierarchical = np.array(m_entropy_hierarchical)

    if normalizer == "arithmetic":
        normalizer = 0.5 * (m_entropy + c_entropy)
        normalizer_hierarchical = 0.5 * (m_entropy_hierarchical + c_entropy_hierarchical)
    elif normalizer == "joint":
        normalizer = joint_mc_entropy
        normalizer_hierarchical = joint_entropy_hierarchical
    else:
        raise AttributeError("Unknown normalizer")

    # normalized mutual information
    normalized_MI = (m_entropy + c_entropy - joint_mc_entropy) / normalizer
    normalized_MI_hierarchical = ((m_entropy_hierarchical + c_entropy_hierarchical - joint_entropy_hierarchical)
                                  / normalizer_hierarchical)

    # normalized version of h(c|m)
    normalized_effectiveness = (joint_mc_entropy - m_entropy) / c_entropy
    normalized_effectiveness_hierarchical = ((joint_entropy_hierarchical - m_entropy_hierarchical) 
                                             / c_entropy_hierarchical)

    # normalized version of h(m|c)
    normalized_consistency = (joint_mc_entropy - c_entropy) / m_entropy
    normalized_consistency_hierarchical = (joint_entropy_hierarchical - c_entropy_hierarchical) / m_entropy_hierarchical

    score_dict = {'normalized_mutual_info': normalized_MI,
                  'normalized_mutual_info_hierarchical': normalized_MI_hierarchical,
                  'effectiveness': 1 - normalized_effectiveness,
                  'effectiveness_hierarchical': 1 - normalized_effectiveness_hierarchical,
                  'consistency': 1 - normalized_consistency,
                  'consistency_hierarchical': 1 - normalized_consistency_hierarchical}

    return score_dict


def cooccurrence_per_hierarchy_level(interaction, n_attributes, n_values, vs_factor):

    vocab_size = (n_values + 1) * vs_factor + 1

    messages = interaction.message.argmax(dim=-1)
    messages = messages[:, :-1].numpy()
    sender_input = interaction.sender_input.numpy()
    relevance_vectors = sender_input[:, -n_attributes:]

    cooccurrence = np.zeros((vocab_size, n_attributes))

    for s in range(vocab_size):
        for i, m in enumerate(messages):
            relevance = relevance_vectors[i]
            cooccurrence[s, int(sum(relevance))] += list(m).count(s)

    cooccurrence = cooccurrence[1:, :]  # remove eos symbol
    split_indices = np.array([np.sum(sender_input[:, -n_attributes:], axis=1) == i for i in range(n_attributes)])
    normalization = np.array([np.sum(split_indices[i]) for i in range(n_attributes)])
    cooccurrence = cooccurrence / normalization

    return cooccurrence


def message_length_per_hierarchy_level(interaction, n_attributes):

    message = interaction.message.argmax(dim=-1)
    relevance_vector = interaction.sender_input[:, -n_attributes:]

    ml_hierarchical = MessageLengthHierarchical.compute_message_length_hierarchical(message, relevance_vector)
    return ml_hierarchical


def symbol_frequency(interaction, n_attributes, n_values, vocab_size, is_gumbel=True):

    messages = interaction.message.argmax(dim=-1) if is_gumbel else interaction.message
    messages = messages[:, :-1]
    sender_input = interaction.sender_input
    k_hots = sender_input[:, :-n_attributes]
    objects = k_hot_to_attributes(k_hots, n_values)
    intentions = sender_input[:, -n_attributes:]  # (0=same, 1=any)

    objects[intentions == 1] = np.nan

    objects = objects
    messages = messages
    favorite_symbol = {}
    mutual_information = {}
    for att in range(n_attributes):
        for val in range(n_values):
            object_labels = (objects[:, att] == val).astype(int)
            max_MI = 0
            for symbol in range(vocab_size):
                symbol_indices = np.argwhere(messages == symbol)[0]
                symbol_labels = np.zeros(len(messages))
                symbol_labels[symbol_indices] = 1
                MI = normalized_mutual_info_score(symbol_labels, object_labels)
                if MI > max_MI:
                    max_MI = MI
                    max_symbol = symbol
            favorite_symbol[str(att) + str(val)] = max_symbol
            mutual_information[str(att) + str(val)] = max_MI

    sorted_objects = []
    sorted_messages = []
    for i in reversed(range(n_attributes)):
        sorted_objects.append(objects[np.sum(np.isnan(objects), axis=1) == i])
        sorted_messages.append(messages[np.sum(np.isnan(objects), axis=1) == i])

    # from most concrete to most abstract (all same to only one same)
    att_val_frequency = np.zeros(n_attributes)
    symbol_frequency = np.zeros(n_attributes)

    for level in range(n_attributes):
        for obj, message in zip(sorted_objects[level], sorted_messages[level]):
            for position in range(len(obj)):
                if not np.isnan(obj[position]):
                    att_val_frequency[level] += 1
                    fav_symbol = favorite_symbol[str(position) + str(int(obj[position]))]
                    symbol_frequency[level] += np.count_nonzero(message == fav_symbol)

    return symbol_frequency / att_val_frequency, mutual_information
