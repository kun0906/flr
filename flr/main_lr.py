"""

"""
import argparse
import copy
import os
import pickle
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils import load_data, evaluate

# ignore some warnings, especially for lbgfs optimization
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='credit_score')
parser.add_argument("--n_repeats", type=int, default=2)  # number of repeats of the experiments
args = parser.parse_args()
print(args)
DATA_NAME = args.data_name
N_REPEATS = args.n_repeats
# N_REPEATS = 2   # number of repeats of the experiments
MAX_ITERS = 100  # maximum iterations of each experiments
OUT_DIR = 'out_baseline'

# def set_model_params(model, params):
#     model = copy.deepcopy(model)
#     model.coef_ = params['coef_']
#     model.intercept_ = params['intercept_']
#     return model


def single_main(data_name, random_state=42):
    (X_train, y_train), (X_test, y_test) = load_data(data_name, random_state=random_state)
    is_normalization = True
    if is_normalization:
        std = StandardScaler()
        std.fit(X_train)
        X_train = std.transform(X_train)
        X_test = std.transform(X_test)

    multi_class = 'ovr' if len(set(y_train)) > 2 else 'auto'
    model = LogisticRegression(max_iter=MAX_ITERS,
                               warm_start=False,
                               multi_class = multi_class,
                               fit_intercept=False,
                               class_weight='balanced',
                               random_state=random_state)
    model.fit(X_train, y_train)

    scores = evaluate(model, X_test, y_test)
    params = {'coef_': model.coef_, 'intercept_': model.intercept_}
    history = {'scores': scores, 'params': params}

    return history


def main():
    history = {}
    # data_name = 'credit_risk'   #, 'bank_marketing', 'credit_score', 'credit_risk'
    for i in range(N_REPEATS):
        random_state = i * 100
        his = single_main(DATA_NAME, random_state=random_state)
        # print(i, his)
        history[i] = his

    out_f = f'{OUT_DIR}/{DATA_NAME}-LR-R_{N_REPEATS}-M_{MAX_ITERS}.dat'
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
            scores.append(his['scores'][metric])
            parmas.append(his['params'])
            print(f'i_repeat:{i_repeat}, {metric}:', scores[-1], parmas[-1])
        print('LR:', metric, np.mean(scores), np.std(scores), scores)


if __name__ == '__main__':
    main()
