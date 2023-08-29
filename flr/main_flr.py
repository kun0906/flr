"""

"""
import collections
import copy
import os
import pickle
import warnings

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils import load_data, evaluate

# ignore some warnings, especially for lbgfs optimization
warnings.filterwarnings('ignore')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='credit_risk')    # credit_risk
parser.add_argument("--n_repeats", type=int, default=2)  # number of repeats of the experiments
parser.add_argument("--agg_method", type=str, default='trim_mean')  # aggregation method for federated learning
parser.add_argument("--percent_adversary", type=float, default=0.2)  # percentage of adversary machines
parser.add_argument("--n_clients", type=int, default=50)  # percentage of adversary machines
parser.add_argument("--part_method", type=str, default='noniid')  # partition method: iid or non-iid
parser.add_argument("--n_i", type=int, default=200)  # number of points per class
args = parser.parse_args()
print(args)
DATA_NAME=args.data_name
N_REPEATS = args.n_repeats
AGG_METHOD = args.agg_method    # aggregation method for federated learning
# PERCENT_ADVERSARY= True if args.percent_adversary == 'true' else False
PERCENT_ADVERSARY= float(args.percent_adversary)
PART_METHOD = args.part_method
MAX_ITERS = 100   # maximum iterations of each experiments
N_I = args.n_i    # number of points per class
N_CLIENTS = args.n_clients  # total number of clients. If N_CLIENTS < 10, the trim_mean doesn't work.
N_Attackers = int(np.floor(PERCENT_ADVERSARY*N_CLIENTS))
N_CLIENTS = N_CLIENTS - N_Attackers  # int is round down.
print(f"N_CLIENTS:{N_CLIENTS}, N_Attackers:{N_Attackers}, M:{MAX_ITERS}, P:{PART_METHOD}, N_i: {N_I}")
# def partition(X: np.ndarray, y: np.ndarray, num_partitions: int):
#     """Split X and y into a number of partitions."""
#     return list(
#         zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
#     )

def iid_partition(X, y, num_partitions, random_state):

    # skf=StratifiedKFold(n_splits=num_partitions, random_state=random_state, shuffle=True)
    # parts = []
    # for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    #     parts.append((X[test_index], y[test_index]))
    # return parts

    rng = np.random.RandomState(seed=random_state)
    # Get unique class labels and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    parts = []
    # Iterate through unique classes and create subsets
    for index, (cnt, class_label) in enumerate(zip(class_counts, unique_classes)):
        class_indices = np.where(y == class_label)[0]
        rng.shuffle(class_indices)  # Shuffle indices
        X_, y_ = X[class_indices], y[class_indices]
        n_i = min(N_I, len(y_)//num_partitions)    # each client has almost equal number of points per class
        i = 0
        while i < num_partitions:
            st = i * n_i
            ed = (i + 1) * n_i
            if index == 0:
                parts.append((X_[st:ed], y_[st:ed]))
            else:
                tmp_X = np.concatenate([parts[i][0], X_[st:ed]], axis=0)
                tmp_y = np.concatenate([parts[i][1], y_[st:ed]])
                parts[i] = (tmp_X, tmp_y)
            i = i + 1

    # print(f'total rows: {sum([len(v[1]) for v in parts])}')
    return parts


def noniid_partition(X, y, num_partitions, random_state):
    """ Non-IID data: each client have all the class data. each client has N_I point from a class and 10% other points
    from the rest classes.
    This scripts might include duplicates during to the sampling on th 10% data from other classes

    :param X:
    :param y:
    :param n_i: each client should have n_i points for each class
    :param random_state:
    :return:
    """

    rng = np.random.RandomState(seed=random_state)
    # Get unique class labels and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    parts = []
    # Iterate through unique classes and create subsets
    left_partitions = num_partitions
    for index, (cnt, class_label) in enumerate(zip(class_counts, unique_classes)):
        class_indices = np.where(y == class_label)[0]
        rng.shuffle(class_indices)  # Shuffle indices
        X_, y_ = X[class_indices], y[class_indices]
        n_i = min(N_I, len(y_) // num_partitions)  # each client has almost equal number of points per class
        if index == len(unique_classes)-1:  # for the last class, we use all the rest partitions
            num_partitions_i = left_partitions
        else:
            num_partitions_i = num_partitions // len(unique_classes)
            left_partitions = left_partitions - num_partitions_i
        start_index = len(parts)
        i = 0
        while i < num_partitions_i:
            st = i * n_i
            ed = (i + 1) * n_i
            parts.append((X_[st:ed], y_[st:ed]))
            # sampling 10% data from other classes
            tmp_X = parts[-1][0]    # be careful of the index here
            tmp_y = parts[-1][1]
            for class_label2 in unique_classes:
                if class_label2 == class_label: continue
                class_indices2 = np.where(y == class_label2)[0]
                rng.shuffle(class_indices2)  # Shuffle indices
                n2_i = int(np.floor(n_i*0.1))   # sampling 10% data from other classes
                class_indices2 = class_indices2[:n2_i]
                X2_, y2_ = X[class_indices2], y[class_indices2]
                tmp_X = np.concatenate([tmp_X, X2_], axis=0)
                tmp_y = np.concatenate([tmp_y, y2_])
            parts[-1] = (tmp_X, tmp_y)       # be careful of the index here
            i = i + 1

    print(f'total rows: {sum([len(v[1]) for v in parts])}')
    return parts

def set_model_params(model, params):
    model = copy.deepcopy(model)
    model.coef_ = params['coef_']
    model.intercept_ = params['intercept_'] # use the default values: 0
    return model


def aggregate_params(params, method='mean', method_params = {}, fit_intercept=True):
    for p in ['coef_', 'intercept_']:
        if method == 'mean':
            # params['coef_'] = np.mean(np.asarray(params['coef_']), axis=0)
            # params['intercept_'] = np.mean(np.asarray(params['intercept_']), axis=0)
            params[p] = np.mean(np.asarray(params[p]), axis=0)
        elif method == 'median':
            params[p] = np.median(np.asarray(params[p]), axis=0)
        elif method == 'trim_mean':
            from scipy import stats
            proportiontocut = method_params['proportiontocut']
            params[p] = stats.trim_mean(np.asarray(params[p]), proportiontocut, axis=0)
        else:
            raise NotImplementedError(f'{method} is not implemented yet.')
    if not fit_intercept:
        params['intercept_'] = np.zeros(params['intercept_'].shape)
    return params


def single_main(data_name, random_state=42):
    print(f'{int(random_state/100)}th repeat, and random_state: {random_state}')
    (X_train, y_train), (X_test, y_test) = load_data(data_name, random_state=random_state)
    print(f'X_train: {X_train.shape}, y_train: {collections.Counter(y_train)}')
    print(f'X_test: {X_test.shape}, y_test: {collections.Counter(y_test)}')
    is_normalization = True
    if is_normalization:
        std = StandardScaler()
        std.fit(X_train)
        X_train = std.transform(X_train)
        X_test = std.transform(X_test)

    N, D = X_train.shape
    history = []
    for iter in tqdm(range(MAX_ITERS + 1), disable=False, desc=f'{random_state//100}th', miniters=MAX_ITERS//10, leave=True):
        if iter == 0:  # initialization
            fit_intercept = False
            # Server: doesn't use fit(), it only aggregates the parameters collected from the clients
            n_classes = len(set(y_train))
            multi_class = 'ovr' if n_classes > 2 else 'auto'
            model = LogisticRegression(multi_class=multi_class, random_state=random_state,
                                       class_weight='balanced',
                                       fit_intercept=fit_intercept)
            model.classes_ = np.unique(y_test)
            server = {'model': model, 'data': (None, None, X_test, y_test), 'scores': None}

            # Clients
            # Split train set into 10 partitions and randomly use one for training.
            # rng = np.random.RandomState(seed=random_state)
            # idx = rng.permutation(len(X_train))
            # X_train, y_train = X_train[idx], y_train[idx]
            # parts = partition(X_train, y_train, N_CLIENTS)
            if PART_METHOD != 'iid':
                parts = noniid_partition(X_train, y_train, N_CLIENTS, random_state)
            else:
                parts = iid_partition(X_train, y_train, N_CLIENTS, random_state)

            clients = {}
            for i_client in range(N_CLIENTS):
                # seed = i_client * 100   # each client has his own seed vs. all the client has the same seed
                _X, _y = parts[i_client]
                data = (_X, _y, X_test, y_test)
                client = {'model': LogisticRegression(max_iter=10,  # local update with 10 iterations
                                                      warm_start=True,
                                                      multi_class= multi_class,
                                                      fit_intercept=fit_intercept,
                                                      class_weight='balanced',
                                                      random_state=random_state),
                          'data': data, 'scores': None}
                clients[i_client] = client

            params = {'coef_': np.zeros((n_classes, D)), 'intercept_': np.zeros((n_classes,))}
        else:
            # Clients training
            for i_client in range(N_CLIENTS):
                client = clients[i_client]
                model = client['model']
                model = set_model_params(model, params)  # set model with the server params
                model.warm_start = True

                X_train, y_train, X_test, y_test = client['data']
                model.fit(X_train, y_train)
                scores = evaluate(model, X_test, y_test)
                clients[i_client]['model'] = model  # update the model
                clients[i_client]['scores'] = scores

            # Server:
            # 1. Params Aggregation
            params = {'coef_': [], 'intercept_': []}
            for i_client in range(N_CLIENTS):
                model = clients[i_client]['model']
                params['coef_'].append(model.coef_)
                params['intercept_'].append(model.intercept_)

            rng_ = np.random.RandomState(seed=iter)
            for _ in range(N_Attackers):
                # adversary: here we use one, and there could be multi-attackers
                coef_ = np.zeros((n_classes, D))
                intercept_ = np.zeros((n_classes, ))
                idx = rng_.choice(range(D)) # random select one coordinate and modify it to a big value.
                coef_[:, idx] = coef_[:,idx] + 1000  # only one coordinate is very large
                intercept_[:,] = intercept_[:,] + 1000
                params['coef_'].append(coef_)
                params['intercept_'].append(intercept_)

            if AGG_METHOD == 'trim_mean':
                method_params = {'proportiontocut': PERCENT_ADVERSARY/2}
            else:
                method_params = {}
            params = aggregate_params(params, method=AGG_METHOD, method_params=method_params,
                                      fit_intercept=fit_intercept)
            # print(iter, params)

            # 2. Evaluation
            model = server['model']
            model = set_model_params(model, params)  # set model with the server params
            X_train, y_train, X_test, y_test = server['data']
            server['scores'] = evaluate(model, X_test, y_test)

        history.append({'server': {'scores': server['scores'], 'params': params}, 'clients': None})

    return history


def main():
    history = {}
    for i in tqdm(range(N_REPEATS), disable=False, desc='N_REPEATS'):
        random_state = i * 100
        his = single_main(DATA_NAME, random_state=random_state)
        # print(i, his)
        history[i] = his[-1]

    out_f = f'out/{DATA_NAME}-FLR-R_{N_REPEATS}-M_{MAX_ITERS}-C_{args.n_clients}-G_{AGG_METHOD}-' \
            f'A_{PERCENT_ADVERSARY}-P_{PART_METHOD}-I_{N_I}.dat'
    os.makedirs(os.path.dirname(out_f), exist_ok=True)
    with open(out_f, 'wb') as f:
        pickle.dump(history, f)

    # format results
    with open(out_f, 'rb') as f:
        history = pickle.load(f)

    for metric in ['accuracy', 'f1', 'auc', 'loss']:
        scores = []
        parmas = []
        for i_repeat, his in history.items():
            scores.append(his['server']['scores'][metric])
            parmas.append(his['server']['params'])
            print(f'i_repeat:{i_repeat}, {metric}:', scores[-1], parmas[-1])
        print('FLR:', metric, np.mean(scores), np.std(scores), scores)


if __name__ == '__main__':
    main()
