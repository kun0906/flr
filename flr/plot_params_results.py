import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np


def extact_data(file_path, metric='coef_'):
    # format results
    with open(file_path, 'rb') as f:
        history = pickle.load(f)

    scores = []
    params = []
    params2 = []
    if 'FLR' in file_path:
        M, D = history[0]['server']['params'][metric].shape
    else:
        M, D = history[0][params][metric].shape
    for i_model in range(M):
        tmp = []
        for i_repeat, his in history.items():
            if 'FLR' in file_path:
                # scores.append(his['server']['scores'][metric])
                params_ =  his['server']['params'][metric]
            else:  # centralized results
                # scores.append(his['scores'][metric])
                params_ =  his['params'][metric]
            tmp.append(params_)
            # print(f'i_repeat:{i_repeat}, {metric}:', scores[-1], params[-1])
        tmp = np.asarray(tmp)[:, i_model, :]
        # params.append((np.mean(tmp, axis=0), np.std(tmp, axis=0)))
        params.append([f"{mu_:.2f} $\pm$ {std_:.2f}" for mu_, std_ in zip(np.mean(tmp, axis=0).flatten(), np.std(tmp, axis=0).flatten())])
        params2.append([(f"{mu_:.2f}", f"{std_:.2f}") for mu_, std_ in
                       zip(np.mean(tmp, axis=0).flatten(), np.std(tmp, axis=0).flatten())])
    # print(f'{file_path}:', metric, f"{np.mean(scores):.2f} $\pm$ {1.96*np.std(scores):.2f}", np.mean(scores), 1.96*np.std(scores), scores)

    return params, params2


def baseline(out_dir, datasets, N_REPEATS=10):

    # PART_METHOD = 'iid'
    for metric in ['accuracy', 'f1', 'auc', 'loss']:
        print('\n', metric)
        for data_name in datasets: #['credit_risk']:  # credit_risk, 'credit_score',
            results = {}
            for alg_name in ['LR', 'DT']:
                if alg_name == 'LR':
                    file_path = os.path.join(out_dir,
                                             f'{data_name}-{alg_name}-R_{N_REPEATS}-M_100.dat')
                else:
                    file_path = os.path.join(out_dir,
                                             f'{data_name}-{alg_name}-R_{N_REPEATS}.dat')
                _mu, _std = extact_data(file_path, metric)
                X, Y, Y_errs = None, _mu, _std
                results[alg_name] = (X, Y, Y_errs)

def plot_feature_importance(X, Y, Y_errs, feature_names, out_file='.png', title=''):
    X, Y, Y_errs, feature_names= np.asarray(X), np.asarray(Y), np.asarray(Y_errs), np.asarray(feature_names)
    import pandas as pd
    import matplotlib.pyplot as plt

    # Plot the feature importances of the forest
    fig, ax = plt.subplots()
    y_pos = X
    hbars = ax.barh(y_pos, Y, xerr=Y_errs, align='center')
    ax.set_yticks(y_pos, labels=feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Coefficient ($\mu \pm \sigma$)')
    # ax.set_title(title)

    # Label with specially formatted floats
    ax.bar_label(hbars, fmt='%.2f')
    ax.set_xlim(left=min(Y)-0.2, right=max(Y)+0.2)  # adjust xlim to fit labels

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(out_file)
    plt.show()



def baseline_FLR(out_dir, datasets, PART_METHOD, N_CLIENTS = 10, N_I=-1):
    # for file_name in [f'{data_name}-FLR-R_10-M_{M}-C_{N_CLIENTS}-G_mean-A_{p}.dat',
    #                   f'{data_name}-FLR-R_10-M_{M}-C_{N_CLIENTS}-G_median-A_{p}.dat',
    #                   f'{data_name}-FLR-R_10-M_{M}-C_{N_CLIENTS}-G_trim_mean-A_{p}.dat',
    #                   # f'{data_name}-LR-R_10-M_50.dat',
    #                   # f'{data_name}-DT-R_10.dat',
    #                   ]:

    # # out_dir = 'out_10000points_each_class'
    # # out_dir = 'out-R_10-20230822'
    # # out_dir = 'out_baseline-R_10-20230827'
    # out_dir = 'out-selected_features-20230829'
    # PART_METHOD = 'iid'
    for metric in ['coef_']:
        print('\n', metric)
        for data_name in datasets: #[ 'credit_risk']:   # credit_risk, 'credit_score',
            results = {}
            for agg_method in ['mean', 'median', 'trim_mean']:
                X = []
                Y = []
                Y_errs = []
                for p in [0.0]:
                    M = 100
                    # N_CLIENTS = 50  # total number of clients. If N_CLIENTS < 10, the trim_mean doesn't work.
                    N_CLIENTS_tmp = N_CLIENTS - int(np.floor(p * N_CLIENTS))
                    N_Attackers = N_CLIENTS - N_CLIENTS_tmp
                    # N_CLIENTS = N_CLIENTS_tmp
                    print(f"N_CLIENTS:{N_CLIENTS}, N_Attackers:{N_Attackers}")
                    # R: number of repeats
                    # M: number of iterations
                    # C: number of clients
                    # G: aggregation method used by the server
                    # A: has adversary?
                    file_path = os.path.join(out_dir,
                                             f'{data_name}-FLR-R_10-M_{M}-C_{N_CLIENTS}-G_{agg_method}-A_{p}'
                                             f'-P_{PART_METHOD}-I_{N_I}.dat')
                    _parmas, _parmas2 = extact_data(file_path, metric)
                    X.append(p)
                    # Y.append(_mu)
                    # Y_errs.append(_std)
                    for _j in range(len(_parmas2)):
                        tmp = sorted([(_idx, float(_mu), float(_std)) for _idx, (_mu, _std) in enumerate(_parmas2[_j])], key=lambda vs:abs(vs[1]), reverse=True)
                        tmp_str = [(_idx, f"{_mu} $\pm$ {_std}") for _idx, _mu, _std in tmp]
                        print(metric, data_name, agg_method, p, f'class_{_j}', tmp_str)

                        # plot feature importance for each model for multiclassification task
                        _idx, _mu, _std = zip(*tmp)
                        _idx = np.asarray(_idx)
                        if data_name == 'credit_score':
                            feature_names = ["Age", "Annual_Income", "Num_Bank_Accounts", "Num_Credit_Card",
                                             "Interest_Rate",
                                             "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
                                             "Changed_Credit_Limit",
                                             "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
                                             "Total_EMI_per_month",
                                             "Amount_invested_monthly", "Monthly_Balance"]
                        elif data_name == 'bank_marketing':
                            feature_names = ["age","balance","day","duration","campaign","pdays","previous"]
                        elif data_name == 'loan_prediction':
                            feature_names = ["Income","Age","Experience","CURRENT_JOB_YRS","CURRENT_HOUSE_YRS"]
                        else:
                            feature_names = _idx
                            # raise NotImplementedError()
                        feature_names = np.asarray(feature_names)[_idx]
                        plot_feature_importance(range(len(_idx)), _mu, _std, feature_names,
                                                out_file=os.path.splitext(file_path)[0] + f'-Class_{_j}.png',
                                                title = f'{data_name}, class_{_j}')
                results[agg_method] = (X, Y, Y_errs)



            # # plot
            # out_file = f"{out_dir}/{metric}-{data_name}-C_{N_CLIENTS}-I_{N_I}-p-{PART_METHOD}.png"
            # markers = ['*', '^', 'o', 'v', '+', '<', '>', '.', ',']
            # colors = ['b', 'purple', 'c', 'y', 'm', 'tab:orange', 'g', ]
            # ecolors = ['tab:red', 'tab:cyan', 'tab:olive', 'tab:green', 'tab:purple', 'tab:brown', ]
            # for _i, agg_method in enumerate(['mean', 'median', 'trim_mean']):
            #     alg_label = f"FLR-{agg_method}"
            #     X, Y, Y_errs = results[agg_method]
            #     Y_errs = [1.96*(v/(len(Y_errs)**0.5)) for v in Y_errs]
            #     plt.errorbar(X, Y, yerr=Y_errs, marker=markers[_i], color=colors[_i],
            #                  capsize=3, ecolor=ecolors[_i],
            #                  markersize=7, markerfacecolor='black',
            #                  label=f"{alg_label}", alpha=1)
            # plt.title(out_file)
            # plt.ylabel(metric)
            # plt.xlabel(f'P (N={N_CLIENTS})')
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(out_file, dpi=300)
            # plt.show()



def main_fixed_n(out_dir, datasets, N_CLIENTS = 5, N_I=100):
    # for file_name in [f'{data_name}-FLR-R_10-M_{M}-C_{N_CLIENTS}-G_mean-A_{p}.dat',
    #                   f'{data_name}-FLR-R_10-M_{M}-C_{N_CLIENTS}-G_median-A_{p}.dat',
    #                   f'{data_name}-FLR-R_10-M_{M}-C_{N_CLIENTS}-G_trim_mean-A_{p}.dat',
    #                   # f'{data_name}-LR-R_10-M_50.dat',
    #                   # f'{data_name}-DT-R_10.dat',
    #                   ]:

    # out_dir = 'out_10000points_each_class'
    # out_dir = 'out-R_10-20230822'
    # out_dir = 'out-R_10-20230827'
    # PART_METHOD = 'iid'
    for metric in ['accuracy', 'f1', 'auc', 'loss']:
        print('\n', metric)
        for data_name in datasets: #['credit_score']:   # credit_risk
            results = {}
            for agg_method in ['mean', 'median', 'trim_mean']:
                X = []
                Y = []
                Y_errs = []
                for p in [0.0, 0.05, 0.1, 0.15, 0.2]:
                    M = 100
                    # N_CLIENTS = 50  # total number of clients. If N_CLIENTS < 10, the trim_mean doesn't work.
                    N_CLIENTS_tmp = N_CLIENTS - int(np.floor(p * N_CLIENTS))
                    N_Attackers = N_CLIENTS - N_CLIENTS_tmp
                    # N_CLIENTS = N_CLIENTS_tmp
                    print(f"N_CLIENTS:{N_CLIENTS}, N_Attackers:{N_Attackers}")
                    # R: number of repeats
                    # M: number of iterations
                    # C: number of clients
                    # G: aggregation method used by the server
                    # A: has adversary?
                    file_path = os.path.join(out_dir,
                                             f'{data_name}-FLR-R_10-M_{M}-C_{N_CLIENTS}-G_{agg_method}-A_{p}'
                                             f'-P_{PART_METHOD}-I_{N_I}.dat')
                    _mu, _std = extact_data(file_path, metric)
                    X.append(p)
                    Y.append(_mu)
                    Y_errs.append(_std)

                results[agg_method] = (X, Y, Y_errs)

            # plot
            out_file = f"{out_dir}/{metric}-{data_name}-C_{N_CLIENTS}-I_{N_I}-p-{PART_METHOD}.png"
            markers = ['*', '^', 'o', 'v', '+', '<', '>', '.', ',']
            colors = ['b', 'purple', 'c', 'y', 'm', 'tab:orange', 'g', ]
            ecolors = ['tab:red', 'tab:cyan', 'tab:olive', 'tab:green', 'tab:purple', 'tab:brown', ]
            for _i, agg_method in enumerate(['mean', 'median', 'trim_mean']):
                alg_label = f"FLR-{agg_method}"
                X, Y, Y_errs = results[agg_method]
                Y_errs = [1.96*(v/(len(Y_errs)**0.5)) for v in Y_errs]
                plt.errorbar(X, Y, yerr=Y_errs, marker=markers[_i], color=colors[_i],
                             capsize=3, ecolor=ecolors[_i],
                             markersize=7, markerfacecolor='black',
                             label=f"{alg_label}", alpha=1)
            plt.title(out_file)
            plt.ylabel(metric)
            plt.xlabel(f'P (N={N_CLIENTS})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_file, dpi=300)
            plt.show()


def main_fixed_p(out_dir, datasets, p = 0.2, N_I=100):

    # out_dir = 'out_10000points_each_class'
    # out_dir = 'out-R_10-20230822'
    # out_dir = 'out-R_10-20230827'
    # PART_METHOD='iid'
    for metric in ['accuracy', 'f1', 'auc', 'loss']:
        print('\n', metric)
        for data_name in datasets: #['credit_score']:   # credit_risk
            results = {}
            for agg_method in ['mean', 'median', 'trim_mean']:
                X = []
                Y = []
                Y_errs = []
                for N_CLIENTS in [10, 50, 100, 150, 200]:
                    M = 100
                    # N_CLIENTS = 50  # total number of clients. If N_CLIENTS < 10, the trim_mean doesn't work.
                    N_CLIENTS_tmp = N_CLIENTS - int(np.floor(p * N_CLIENTS))
                    N_Attackers = N_CLIENTS - N_CLIENTS_tmp
                    # N_CLIENTS = N_CLIENTS_tmp
                    print(f"N_CLIENTS:{N_CLIENTS}, N_Attackers:{N_Attackers}")
                    # R: number of repeats
                    # M: number of iterations
                    # C: number of clients
                    # G: aggregation method used by the server
                    # A: has adversary?
                    file_path = os.path.join(out_dir,
                                             f'{data_name}-FLR-R_10-M_{M}-C_{N_CLIENTS}-G_{agg_method}-A_{p}'
                                              f'-P_{PART_METHOD}-I_{N_I}.dat')
                    _mu, _std = extact_data(file_path, metric)
                    X.append(N_CLIENTS)
                    Y.append(_mu)
                    Y_errs.append(_std)

                results[agg_method] = (X, Y, Y_errs)

            # plot
            out_file = f"{out_dir}/{metric}-{data_name}-p_{p}-I_{N_I}-C-{PART_METHOD}.png"
            markers = ['*', '^', 'o', 'v', '+', '<', '>', '.', ',']
            colors = ['b', 'purple', 'c', 'y', 'm', 'tab:orange', 'g', ]
            ecolors = ['tab:red', 'tab:cyan', 'tab:olive', 'tab:green', 'tab:purple', 'tab:brown', ]
            for _i, agg_method in enumerate(['mean', 'median', 'trim_mean']):
                alg_label = f"FLR-{agg_method}"
                X, Y, Y_errs = results[agg_method]
                Y_errs = [1.96*(v/(len(Y_errs)**0.5)) for v in Y_errs]
                plt.errorbar(X, Y, yerr=Y_errs, marker=markers[_i], color=colors[_i],
                             capsize=3, ecolor=ecolors[_i],
                             markersize=7, markerfacecolor='black',
                             label=f"{alg_label}", alpha=1)
            plt.title(out_file)
            plt.ylabel(metric)
            plt.xlabel(f'N (p={p})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_file, dpi=300)
            plt.show()



def main_varied_ni(out_dir, datasets, p = 0.2, N_CLIENTS=100):

    # out_dir = 'out_10000points_each_class'
    # out_dir = 'out-R_10-20230822'
    # out_dir = 'out-R_10-20230827'
    # PART_METHOD='iid'
    for metric in ['accuracy', 'f1', 'auc', 'loss']:
        print('\n', metric)
        for data_name in datasets: #['credit_score']:   # credit_risk
            results = {}
            for agg_method in ['mean', 'median', 'trim_mean']:
                X = []
                Y = []
                Y_errs = []
                for N_I in [25, 50, 100, 150, 200]:
                    M = 100
                    # N_CLIENTS = 50  # total number of clients. If N_CLIENTS < 10, the trim_mean doesn't work.
                    N_CLIENTS_tmp = N_CLIENTS - int(np.floor(p * N_CLIENTS))
                    N_Attackers = N_CLIENTS - N_CLIENTS_tmp
                    # N_CLIENTS = N_CLIENTS_tmp
                    print(f"N_CLIENTS:{N_CLIENTS}, N_Attackers:{N_Attackers}")
                    # R: number of repeats
                    # M: number of iterations
                    # C: number of clients
                    # G: aggregation method used by the server
                    # A: has adversary?
                    file_path = os.path.join(out_dir,
                                             f'{data_name}-FLR-R_10-M_{M}-C_{N_CLIENTS}-G_{agg_method}-A_{p}'
                                              f'-P_{PART_METHOD}-I_{N_I}.dat')
                    _mu, _std = extact_data(file_path, metric)
                    X.append(N_I)
                    Y.append(_mu)
                    Y_errs.append(_std)

                results[agg_method] = (X, Y, Y_errs)

            # plot
            out_file = f"{out_dir}/{metric}-{data_name}-p_{p}-C_{N_CLIENTS}-I-{PART_METHOD}.png"
            markers = ['*', '^', 'o', 'v', '+', '<', '>', '.', ',']
            colors = ['b', 'purple', 'c', 'y', 'm', 'tab:orange', 'g', ]
            ecolors = ['tab:red', 'tab:cyan', 'tab:olive', 'tab:green', 'tab:purple', 'tab:brown', ]
            for _i, agg_method in enumerate(['mean', 'median', 'trim_mean']):
                alg_label = f"FLR-{agg_method}"
                X, Y, Y_errs = results[agg_method]
                Y_errs = [1.96*(v/(len(Y_errs)**0.5)) for v in Y_errs]
                plt.errorbar(X, Y, yerr=Y_errs, marker=markers[_i], color=colors[_i],
                             capsize=3, ecolor=ecolors[_i],
                             markersize=7, markerfacecolor='black',
                             label=f"{alg_label}", alpha=1)
            plt.title(out_file)
            plt.ylabel(metric)
            plt.xlabel(f'n_i (p={p})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_file, dpi=300)
            plt.show()


if __name__ == '__main__':

    # out_dir = 'out_baseline-R_10-20230827'
    # out_dir = 'out_baseline_selected_features-20230829'
    out_dir = 'out-20230902/baseline'
    datasets = ['credit_score', ] # 'bank_marketing', 'loan_prediction', 'credit_risk'
    # baseline(out_dir, datasets)  # LR, DT
    for PART_METHOD in ['iid', 'noniid']:
        baseline_FLR(out_dir, datasets, PART_METHOD)

    # out_dir = 'out-selected_features-20230829'
    # for PART_METHOD in ['iid', 'noniid']:
    #
    #     # for N_CLIENTS in [100]:
    #     #     main_fixed_n(out_dir, datasets, N_CLIENTS, N_I=100)  # changed p
    #     #
    #     for p in [0.1]:
    #         main_fixed_p(out_dir, datasets, p, N_I=100) # changed n_clients
    #
    #     # for p in [0.1]:  # various n_i, fixed p=0.1, n = 100
    #     #     main_varied_ni(out_dir, datasets, p, N_CLIENTS=100)

