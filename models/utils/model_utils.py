import json
import numpy as np
import os
from collections import defaultdict


def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = list(data['x'])
    data_y = list(data['y'])

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

def batch_data_with_weights(data, data_sample_weights, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = list(data['x'])
    data_y = list(data['y'])

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    np.random.set_state(rng_state)
    np.random.shuffle(data_sample_weights)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        batch_weights = data_sample_weights[i:i+batch_size]
        yield (batched_x, batched_y, batch_weights)

def batch_data_binary_oversampling(data, batch_size, seed, label_to_oversample):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)

    Perform oversampling over the

    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = np.array(data['x'])
    data_y = np.array(data['y'])
    # Randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)


    # Oversample the negative class ('0') or the positive class ('1') 
    pos_features = data_x[data_y == 1]
    neg_features = data_x[data_y == 0]

    pos_labels = data_y[data_y == 1]
    neg_labels = data_y[data_y == 0]

    # First, we make sure that both label are present in the dataset
    # if not then we use the batch_data function
    if len(neg_features) == 0 or len(pos_features) == 0:
        return batch_data(data, batch_size, seed)
    else:
        if label_to_oversample == 0: # Class '0' is the minority class
            ids = np.arange(len(neg_features))
            choices = np.random.choice(ids, size = len(pos_features))

            res_neg_features = neg_features[choices]
            res_neg_labels = neg_labels[choices]

            resampled_features = np.concatenate([pos_features, res_neg_features], axis=0)
            resampled_labels = np.concatenate([pos_labels, res_neg_labels], axis=0)
        else:                       # Class '1' is the minority class
            ids = np.arange(len(pos_features))
            choices = np.random.choice(ids, size = len(neg_features))

            res_pos_features = pos_features[choices]
            res_pos_labels = pos_labels[choices]

            resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
            resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

        # Randomly shuffle samples
        order = np.arange(len(resampled_labels))
        np.random.shuffle(order)
        resampled_features = resampled_features[order]
        resampled_labels = resampled_labels[order]

        # loop through mini-batches
        for i in range(0, len(resampled_features), batch_size):
            batched_x = resampled_features[i:i+batch_size]
            batched_y = resampled_labels[i:i+batch_size]
            yield (batched_x, batched_y)

def batch_data_binary_oversampling_with_weights(data, data_sample_weights, batch_size, seed, label_to_oversample):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)

    Perform oversampling over the

    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = np.array(data['x'])
    data_y = np.array(data['y'])

    # Randomly shuffle data
    np.random.seed(seed)
    order = np.arange(len(data_x))
    np.random.shuffle(order)
    data_x = data_x[order]
    data_y = data_y[order]
    data_sample_weights = data_sample_weights[order]
    
    # Oversample the negative class ('0') or the positive class ('1') 
    pos_features = data_x[data_y == 1]
    neg_features = data_x[data_y == 0]

    pos_labels = data_y[data_y == 1]
    neg_labels = data_y[data_y == 0]

    pos_weights = data_sample_weights[data_y == 1]
    neg_weights = data_sample_weights[data_y == 0]

    # First, we make sure that both label are present in the dataset
    # if not then we use the batch_data function
    if len(neg_features) == 0 or len(pos_features) == 0:
        return batch_data(data, batch_size, seed)
    else:
        if label_to_oversample == 0: # Class '0' is the minority class
            ids = np.arange(len(neg_features))
            choices = np.random.choice(ids, size = len(pos_features))

            res_neg_features = neg_features[choices]
            res_neg_labels = neg_labels[choices]
            res_neg_weights = neg_weights[choices]

            resampled_features = np.concatenate([pos_features, res_neg_features], axis=0)
            resampled_labels = np.concatenate([pos_labels, res_neg_labels], axis=0)
            resampled_sample_weights = np.concatenate([pos_weights, res_neg_weights], axis=0)
        else:                       # Class '1' is the minority class
            ids = np.arange(len(pos_features))
            choices = np.random.choice(ids, size = len(neg_features))

            res_pos_features = pos_features[choices]
            res_pos_labels = pos_labels[choices]
            res_pos_weights = pos_weights[choices]

            resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
            resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)
            resampled_sample_weights = np.concatenate([neg_weights, res_pos_weights], axis=0)

        # Randomly shuffle samples
        order = np.arange(len(resampled_labels))
        np.random.shuffle(order)
        resampled_features = resampled_features[order]
        resampled_labels = resampled_labels[order]
        resampled_sample_weights = resampled_sample_weights[order]

        # loop through mini-batches
        for i in range(0, len(resampled_features), batch_size):
            batched_x = resampled_features[i:i+batch_size]
            batched_y = resampled_labels[i:i+batch_size]
            batch_weights = resampled_sample_weights[i:i+batch_size]
            yield (batched_x, batched_y, batch_weights)

def get_sample_weights(data, sensitive_attribute):
    data_x = np.array(data['x'])
    data_y = np.array(data['y'])

    num_samples = data_y.shape[0]
    sample_weights = np.zeros(num_samples)

    # Count positive/negative and privileged/unprivileged samples
    pos_features = data_x[data_y == 1]
    neg_features = data_x[data_y == 0]

    pos_labels = data_y[data_y == 1]
    neg_labels = data_y[data_y == 0]

    priv_features = data_y[data_x[:,sensitive_attribute] == 1]
    unpriv_features = data_y[data_x[:,sensitive_attribute] == 0]

    # Split samples by labels and sensitive attributes
    
    pos_unpriv_indices = np.equal(pos_features[:,sensitive_attribute],np.zeros(pos_features.shape[0]))
    pos_priv_indices = np.equal(pos_features[:,sensitive_attribute],np.ones(pos_features.shape[0]))
    neg_unpriv_indices = np.equal(neg_features[:,sensitive_attribute],np.zeros(neg_features.shape[0]))
    neg_priv_indices = np.equal(neg_features[:,sensitive_attribute],np.ones(neg_features.shape[0]))

    pos_unpriv_labels = pos_labels[pos_unpriv_indices]
    pos_priv_labels = pos_labels[pos_priv_indices]
    neg_unpriv_labels = neg_labels[neg_unpriv_indices]
    neg_priv_labels = neg_labels[neg_priv_indices]

    cnt_pos, cnt_neg = len(pos_features), len(neg_features)
    cnt_unpriv, cnt_priv = len(unpriv_features), len(priv_features)
    cnt_pos_unpriv, cnt_pos_priv = len(pos_unpriv_labels), len(pos_priv_labels)
    cnt_neg_unpriv, cnt_neg_priv = len(neg_unpriv_labels), len(neg_priv_labels)

    for ind in range(num_samples):
        if data_x[ind,sensitive_attribute] == 0 and data_y[ind] == 0:
            sample_weights[ind] = cnt_neg * cnt_unpriv / (cnt_neg_unpriv * num_samples)
        elif data_x[ind,sensitive_attribute] == 0 and data_y[ind] == 1:
            sample_weights[ind] = cnt_pos * cnt_unpriv / (cnt_pos_unpriv * num_samples)
        elif data_x[ind,sensitive_attribute] == 1 and data_y[ind] == 0:
            sample_weights[ind] = cnt_neg * cnt_priv / (cnt_neg_priv * num_samples)
        elif data_x[ind,sensitive_attribute] == 1 and data_y[ind] == 1:
            sample_weights[ind] = cnt_pos * cnt_priv / (cnt_pos_priv * num_samples)
        
    return sample_weights

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data
