"""

"""
import copy
import os
import pickle
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils import load_credit_risk, evaluate

# ignore some warnings, especially for lbgfs optimization
warnings.filterwarnings('ignore')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_repeats", type=int, default=2)  # number of repeats of the experiments
args = parser.parse_args()
print(args)
N_REPEATS = args.n_repeats
# N_REPEATS = 2   # number of repeats of the experiments
MAX_ITERS = 5   # maximum iterations of each experiments
N_CLIENTS = 100 # number of clients


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )

def set_model_params(model, params):
    model = copy.deepcopy(model)
    model.coef_ = params['coef_']
    model.intercept_ = params['intercept_']
    return model

def single_main(random_state=42):
    (X_train, y_train), (X_test, y_test) = load_credit_risk(random_state=random_state)
    is_normalization = True
    if is_normalization:
        std = StandardScaler()
        std.fit(X_train)
        X_train = std.transform(X_train)
        X_test = std.transform(X_test)

    history = []
    for i in range(MAX_ITERS + 1):
        if i == 0:  # initialization

            # Server
            model = LogisticRegression(random_state=random_state)
            model.classes_ = np.unique(y_test)
            server = {'model': model, 'data': (None, None, X_test, y_test), 'scores': None}

            # Clients
            # Split train set into 10 partitions and randomly use one for training.
            rng = np.random.RandomState(seed=random_state)
            idx = rng.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]
            parts = partition(X_train, y_train, N_CLIENTS)

            clients = {}
            for i_client in range(N_CLIENTS):
                # seed = i_client * 100   # each client has his own seed vs. all the client has the same seed
                _X, _y = parts[i_client]
                data = (_X, _y, X_test, y_test)
                client = {'model': LogisticRegression(max_iter=10,  # local update
                                                      warm_start=True,
                                                      random_state=random_state),
                          'data': data, 'scores': None}
                clients[i_client] = client

            params = {'coef_': np.zeros((1, X_train.shape[-1])), 'intercept_': np.zeros((1,))}
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
            params['coef_'] = np.mean(np.asarray(params['coef_']), axis=0)
            params['intercept_'] = np.mean(np.asarray(params['intercept_']), axis=0)
            # 2. Evaluation
            model = server['model']
            model = set_model_params(model, params)  # set model with the server params
            X_train, y_train, X_test, y_test = server['data']
            server['scores'] = evaluate(model, X_test, y_test)

        history.append({'server': {'scores': server['scores'], 'params': params}, 'clients': None})

    return history


def main():
    history = {}
    for i in range(N_REPEATS):
        random_state = i * 100
        his = single_main(random_state=random_state)
        # print(i, his)
        history[i] = his[-1]

    out_f = f'out/FLR-R_{N_REPEATS}-M_{MAX_ITERS}-C_{N_CLIENTS}.dat'
    os.makedirs(os.path.dirname(out_f), exist_ok=True)
    with open(out_f, 'wb') as f:
        pickle.dump(history, f)

    # format results
    with open(out_f, 'rb') as f:
        history = pickle.load(f)

    for metric in ['loss', 'accuracy']:
        scores = []
        parmas = []
        for i_repeat, his in history.items():
            scores.append(his['server']['scores'][metric])
            parmas.append(his['server']['params'])
            print(f'i_repeat:{i_repeat}, {metric}:', scores[-1], parmas[-1])
        print('FLR:', metric, np.mean(scores), np.std(scores), scores)


if __name__ == '__main__':
    main()
