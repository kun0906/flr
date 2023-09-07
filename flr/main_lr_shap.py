"""

"""
import argparse
import copy
import os
import pickle
import warnings
import shap
# shap.initjs()
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
OUT_DIR = 'out/baseline'

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
    model2 = model.fit(X_train, y_train)

    scores = evaluate(model, X_test, y_test)
    params = {'coef_': model.coef_, 'intercept_': model.intercept_}
    history = {'scores': scores, 'params': params}

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    if len(set(y_train)) == 2:
        # for binary
        # visualize the first prediction's explanation
        shap.plots.waterfall(shap_values[0])

        # visualize the first prediction's explanation with a force plot
        shap.plots.force(shap_values[0])

        # summarize the effects of all the features
        shap.plots.beeswarm(shap_values)

        shap.plots.bar(shap_values)

    else:
        n_classes = len(set(y_test))
        for i in range(n_classes):
            # credit_score:
            feature_names = ["Age","Annual_Income","Num_Bank_Accounts","Num_Credit_Card","Interest_Rate",
                             "Num_of_Loan","Delay_from_due_date","Num_of_Delayed_Payment","Changed_Credit_Limit",
                             "Num_Credit_Inquiries","Outstanding_Debt","Credit_Utilization_Ratio","Total_EMI_per_month",
                             "Amount_invested_monthly","Monthly_Balance"]
            # shap.plots.beeswarm(shap_values[:, :, i], show=False)
            # plt.yticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
            # plt.ylabel("Feature Names")
            # Create the beeswarm plot
            shap.summary_plot(shap_values[:, :, i], feature_names=feature_names, show=False, max_display=10)
            plt.tight_layout()
            plt.savefig(f'{data_name}-{i}.png', bbox_inches='tight', format='png')
            plt.show()

        # # for multiclassification: not work
        # if data_name == 'credit_score':
        #     fig, axes = plt.subplots(nrows=1, ncols=3)
        #     axes = axes.reshape((1, 3))
        # else:
        #     fig, axes = plt.subplots(nrows=2, ncols=3)
        # n_classes = len(set(y_test))
        # for i in range(n_classes):
        #     # visualize the first prediction's explanation
        #     r, c = divmod(i, 3)
        #     # axes[r, c] = shap.plots.waterfall(shap_values[0][:, 0],max_display=X_test.shape[1], show=False)
        #     # plt.tight_layout()
        #     # plt.show()
        #
        #     # visualize the first prediction's explanation with a force plot
        #     # axes[r,c] = shap.plots.force(shap_values[i][:, 0], show=False, matplotlib=False)
        #     # shap.dependence_plot(c, shap_values, X_test, ax=axes[r, c], show=False)
        #     ax0 = fig.add_subplot(1, 3, i+1)
        #     ax0.title.set_text('Class 2 - Best ')
        #     # shap.summary_plot(shap_values[:, :, i], X_test, plot_type="bar", show=False)
        #     ax0.figure = shap.plots.force(shap_values[i][:, 0], show=False)
        #     ax0.set_xlabel(r'SHAP values', fontsize=11)
        #     plt.subplots_adjust(wspace=5)
        #     # plt.tight_layout()
        #     # plt.show()
        #
        #     # summarize the effects of all the features
        #     # axes[r, c] = shap.plots.beeswarm(shap_values[:, :, 0], show=False)
        #     # plt.tight_layout()
        #     # plt.show()
        #
        #     # axes[r, c] = shap.plots.bar(shap_values[:, :, 0], show=False)
        # plt.tight_layout()
        # plt.show()
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
