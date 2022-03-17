import pickle
import numpy as np


def load_accuracies(all_paths, n_runs=5, n_epochs=300, val_steps=10):
    """ loads all accuracies into a dictionary, val_steps should be set to the same as val_frequency during training
    """

    result_dict = {'train_acc': [], 'val_acc': [], 'test_acc': [], 'zs_acc_objects': [], 'zs_acc_abstraction': []}

    for path_idx, path in enumerate(all_paths):

        train_accs = []
        val_accs = []
        zs_accs_objects = []
        zs_accs_abstraction = []

        for run in range(n_runs):
            
            standard_path = path + '/standard/' + str(run) + '/'
            zero_shot_path = path + '/zero_shot/' + str(run) + '/'
            
            # train and validation accuracy
            
            data = pickle.load(open(standard_path + 'loss_and_metrics.pkl', 'rb'))
            lists = sorted(data['metrics_train0'].items())
            _, train_acc = zip(*lists)
            train_accs.append(train_acc)
            lists = sorted(data['metrics_test0'].items()) 
            _, val_acc = zip(*lists)
            if len(val_acc) > n_epochs // val_steps:  # we had some runs where we set val freq to 5 instead of 10
                val_acc = val_acc[::2]
            val_accs.append(val_acc)
            zs_accs_objects.append(data['final_test_acc'])
            
            # zero shot accuracy
            zs_data = pickle.load(open(zero_shot_path + 'loss_and_metrics.pkl', 'rb'))
            zs_accs_abstraction.append(zs_data['final_test_acc'])

        result_dict['train_acc'].append(train_accs)
        result_dict['val_acc'].append(val_accs)
        result_dict['zs_acc_objects'].append(zs_accs_objects)
        result_dict['zs_acc_abstraction'].append(zs_accs_abstraction)

    for key in result_dict.keys():
        result_dict[key] = np.array(result_dict[key])
        
    return result_dict


def load_entropies(all_paths, n_runs=5):
    """ loads all entropy scores into a dictionary"""

    result_dict = {'NI': [], 'effectiveness': [], 'consistency': [],
                   'NI_hierarchical': [], 'effectiveness_hierarchical': [], 'consistency_hierarchical': []}

    for path_idx, path in enumerate(all_paths):

        NIs, effectiveness_scores, consistency_scores = [], [], []
        NIs_hierarchical, effectiveness_scores_hierarchical, consistency_scores_hierarchical = [], [], []

        for run in range(n_runs):

            standard_path = path + '/standard/' + str(run) + '/'
            data = pickle.load(open(standard_path + 'entropy_scores.pkl', 'rb'))
            NIs.append(data['normalized_mutual_info'])
            effectiveness_scores.append(data['effectiveness'])
            consistency_scores.append(data['consistency'])
            NIs_hierarchical.append(data['normalized_mutual_info_hierarchical'])
            effectiveness_scores_hierarchical.append(data['effectiveness_hierarchical'])
            consistency_scores_hierarchical.append(data['consistency_hierarchical'])

        result_dict['NI'].append(NIs)
        result_dict['consistency'].append(consistency_scores)
        result_dict['effectiveness'].append(effectiveness_scores)
        result_dict['NI_hierarchical'].append(NIs_hierarchical)
        result_dict['consistency_hierarchical'].append(consistency_scores_hierarchical)
        result_dict['effectiveness_hierarchical'].append(effectiveness_scores_hierarchical)

    for key in result_dict.keys():
        result_dict[key] = np.array(result_dict[key])

    return result_dict
