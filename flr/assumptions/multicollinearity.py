"""Test multicollinearity

    Main methods:
        1. Correlation Matrix:
        2. Variance Inflation Factor (VIF): 1/(1-R**2)
        3. Tolerance: 1-R**2

    Instruction:
        cd flr
        python3 assumptions/multicollinearity.py



    # threshold = 10

    # bank_marketing
        0th repeat, and random_state: 42
        X_train: (36168, 7), y_train: Counter({0: 31937, 1: 4231})
        X_test: (9043, 7), y_test: Counter({0: 7985, 1: 1058})
        ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'] y
        Selected columns: [0, 1, 2, 3, 4, 5, 6]

    # loan_prediction
        /Users/kun/miniconda3/envs/flr/bin/python /Users/kun/Projects/flr/flr/assumptions/multicollinearity.py
        0th repeat, and random_state: 42
        X_train: (201600, 5), y_train: Counter({0: 176803, 1: 24797})
        X_test: (50400, 5), y_test: Counter({0: 44201, 1: 6199})
        ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS'] Risk_Flag
        Selected columns: [0, 1, 2, 3, 4]

    # CreditScore:
        0th repeat, and random_state: 42
        X_train: (80000, 15), y_train: Counter({1: 42539, 0: 23199, 2: 14262})
        X_test: (20000, 15), y_test: Counter({1: 10635, 0: 5799, 2: 3566})
        ['Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'] Credit_Score
        Selected columns: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # CreditRisk
        /Users/kun/miniconda3/envs/flr/bin/python /Users/kun/Projects/flr/flr/assumptions/multicollinearity.py
        0th repeat, and random_state: 42
        X_train: (704843, 30), y_train: Counter({0: 481423, 1: 166178, 2: 36198, 3: 9273, 4: 6768, 5: 5003})
        X_test: (176211, 30), y_test: Counter({0: 120356, 1: 41545, 2: 9050, 3: 2318, 4: 1692, 5: 1250})
        ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_rv_24m', 'total_rev_hi_lim'] loan_status
        70368744177664.0 [16] 29 [712.9123466313519, 3090.989344294068, 2358.537923135735, 1.8302030302545087, 11.932691897902565, 1.331059791563766, 1.0497452970666488, 1.0597120016519004, 1.146319658223793, 2.1236176043745663, 1.0439916516827819, 4.4625395335906, 1.5987294390484212, 2.1111884953051927, 185948.41873975002, 185983.13653398448, 0.0, 1857.5380674922214, 50039995859672.18, 4990138091269.248, 18117330.49211318, 186743500398.9176, 2.894085511593605, 3.36848056835066, 1.0119180176092708, 1.021207052766473, 1.009097943312865, 1.504705104513247, 1.014113999038074, 4.547130389851461]
        185982.73086821247 [16, 15] 28 [712.8929274130452, 3090.933533972736, 2358.529073974302, 1.8302018195157772, 11.932689678560672, 1.3310596758539754, 1.0497452967150416, 1.0597110041098274, 1.1463193569065595, 2.123613986632737, 1.0439916443862851, 4.462538457245686, 1.5987261379110185, 2.1111810922759813, 185948.0087225266, 0.0, 0.0, 1857.5356253710934, 1312.9059476917096, 134.51342945006454, 1.0246787470223753, 8.418832231985569, 2.894078802781348, 3.3684723113135218, 1.011917546681381, 1.0212055256153592, 1.0090977102265177, 1.5047039956198007, 1.0141129172063643, 4.547129932435405]
        3071.551810666741 [16, 15, 1] 27 [712.8797170995623, 0.0, 2337.853327901423, 1.8212451431838685, 11.932549799473964, 1.3309513599828122, 1.0496814449702805, 1.0597092047033185, 1.1462804214680566, 2.123585655060903, 1.0439673935577924, 4.462167163811498, 1.5987210086347126, 2.1111805863965487, 13.280111622658152, 0.0, 0.0, 1847.346805751096, 1306.1849169570137, 133.64863058963573, 1.0246781467287434, 8.393620822855752, 2.8940714598394313, 3.366393859939594, 1.011892926408061, 1.021200694420018, 1.0090976428951104, 1.504697305231592, 1.0140907562789891, 4.546641358875535]
        692.7750177453909 [16, 15, 1, 17] 26 [540.29813679973, 0.0, 567.8835420480987, 1.82120234465155, 11.83371631898683, 1.3308350967340339, 1.0496805114746766, 1.0596870525948934, 1.1462801728759044, 2.123585588029835, 1.0439496123612366, 4.461756502545064, 1.5987156319220244, 2.111180545238532, 13.279836639541005, 0.0, 0.0, 0.0, 490.60316349982486, 51.63733710074912, 1.0238030364655895, 5.359170891992464, 2.893737529420077, 3.362840159984319, 1.0118910704472357, 1.0211999718763476, 1.0090956240418987, 1.5044204452507903, 1.0140786139290785, 4.546136987588171]
        227.06894991106347 [16, 15, 1, 17, 0] 25 [0.0, 0.0, 226.90953814496146, 1.8211996812558433, 11.744197095741324, 1.3308213749622635, 1.0496779204282987, 1.0596815706040246, 1.1462584817097676, 2.123585206324221, 1.043924594002953, 4.461527935846025, 1.5986886189178087, 2.111166934840747, 13.07805352383746, 0.0, 0.0, 0.0, 11.904504645554423, 2.999633359348877, 1.0237920705388022, 3.73778833645997, 2.8937309875037243, 3.3628381862289745, 1.0118876464377107, 1.0211977269347607, 1.0090930543986383, 1.5043454888009022, 1.014077584465237, 4.545759632476527]
        24.55703500548153 [16, 15, 1, 17, 0, 2] 24 [0.0, 0.0, 0.0, 1.8207182677698177, 11.597954487647055, 1.330299456982905, 1.049589548997177, 1.0593886946141138, 1.1462451293340261, 2.1231556516061953, 1.0433178690959972, 4.4496225726846985, 1.5949528009383114, 2.1104342044642106, 12.915947636078316, 0.0, 0.0, 0.0, 11.809691717360879, 2.989332587311442, 1.0225963469639303, 3.7255105999575355, 2.8937031584327015, 3.362439923774562, 1.0118547985159025, 1.0211915793071862, 1.009028193690976, 1.503711875322078, 1.0140775762412606, 4.537722648697365]
        10.222293020576593 [16, 15, 1, 17, 0, 2, 18] 23 [0.0, 0.0, 0.0, 1.8199325316550712, 6.4285731296031825, 1.330010339734212, 1.0495895321431536, 1.0573274271655488, 1.1408554868749592, 2.1211047958748526, 1.0413154317818567, 4.447056921588152, 1.5940962683475046, 2.1037677140250546, 6.414633060633005, 0.0, 0.0, 0.0, 0.0, 2.7345001916792526, 1.022089216916729, 3.3033030234028224, 2.8607239522747556, 3.1718173157923784, 1.0117021314674022, 1.0211065893825748, 1.0088328190641396, 1.4989905980252318, 1.0139418135120766, 4.5241071092791545]
        Selected columns: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

"""
import collections

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from utils import load_data, get_column_names

import matplotlib.pyplot as plt


# def vif(X):
#     """ VIF measures how much the variance of the estimated regression coefficient of a predictor is
#     increased due to multicollinearity with other predictors.
#     High VIF values indicate strong multicollinearity.
#
#     Generally, if a predictor has a VIF greater than 5 or 10, it indicates potential multicollinearity
#     that should be investigated further.
#
#     for each variable f_i,
#     1. we bulid a linear regression with the rest of features and label (y=f_i)
#     2. r_squared = 1 - SSE/SST = 1 - sum(y_i - y_pred)**2/sum(y_i-mu), where mu = mean(f_i)
#     3. vif_i = 1/(1-r_squared)
#
#     :param X:
#     :return:
#     """
#     # X = np.random.normal(loc=0, scale=1, size=100)
#     from statsmodels.stats.outliers_influence import variance_inflation_factor
#     n, d = X.shape
#     res = [0] * d
#     for j in range(d):
#         res[j] = variance_inflation_factor(X, j)  # OLS(X^', Xj), where X^' doesn't include Xj
#
#     print(f"vif: {res}")
#     print(f"vif (sorted): {sorted(res, reverse=True)}")
#     return res
#

def feature_selection_vif(X, vif_threshold=1):
    """
     # Set a threshold for VIF (e.g., 10)
    VIF measures how much the variance of the estimated regression coefficient of a predictor is
    increased due to multicollinearity with other predictors.
    High VIF values indicate strong multicollinearity.

    Generally, if a predictor has a VIF greater than 5 or 10, it indicates potential multicollinearity
    that should be investigated further.

    for each variable f_i,
        1. we bulid a linear regression with the rest of features and label (y=f_i)
        2. r_squared = 1 - SSE/SST = 1 - sum(y_i - y_pred)**2/sum(y_i-mu), where mu = mean(f_i)
        3. vif_i = 1/(1-r_squared)

    :param X:
    :return:
    """
    # Calculate initial VIF values
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    columns = np.asarray([True for i in range(X.shape[1])])

    # Calculate initial VIF values
    vif = np.array([variance_inflation_factor(X, i) for i in range(X.shape[1])])
    max_vif = max(vif)

    # Perform VIF-based feature elimination iteratively
    removed_features = []
    # We at least keep two features.
    while max_vif >= vif_threshold and sum(columns) > 2:
        # Find the variable with the highest VIF
        is_done = True
        for i, (flg, v) in enumerate(zip(columns, vif)):
            if not flg: continue
            if v == max_vif:
                columns[i] = False
                is_done = False
                vif[i] = 0
                removed_features.append(i)
                break
        if is_done: break
        print(max_vif, removed_features, sum(columns), list(vif))
        # Recalculate VIF values for the rest features
        vif = np.zeros((len(columns),))
        j = 0
        for i, flg in enumerate(columns):
            if not flg: continue
            _X = X[:, columns]
            vif[i] = variance_inflation_factor(_X, j)
            j += 1
        max_vif = max(vif)
        # print(max_vif, sum(columns), list(vif))

    # Display the selected variables
    selected_columns = [i for i in range(X.shape[1]) if columns[i] == True]
    print("Selected columns:", selected_columns)

    return selected_columns


def correlation(df, img_file='.png', title=''):
    corr = df.corr()

    plt.figure(figsize=(18, 15))  # width height
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(img_file, dpi=300, pad_inches=0.0)
    plt.show()


def corr_feat_label(feat, label):
    from scipy.stats.stats import pearsonr
    rho, p = pearsonr(feat, label)
    """
    :return
        statistic : float
            Pearson product-moment correlation coefficient.
        pvalue : float
            The p-value associated with the chosen alternative.
    """
    print(rho, p)
    return rho


if __name__ == '__main__':
    random_state = 42
    for data_name in ['bank_marketing']:  # 'credit_score', loan_prediction, credit_score, bank_marketing
        print(f'{int(random_state / 100)}th repeat, and random_state: {random_state}')
        (X_train, y_train), (X_test, y_test) = load_data(data_name, random_state=random_state, is_selection=False)
        columns, label = get_column_names(data_name, random_state)
        print(f'X_train: {X_train.shape}, y_train: {collections.Counter(y_train)}')
        print(f'X_test: {X_test.shape}, y_test: {collections.Counter(y_test)}')
        print(columns, label)
        is_normalization = True  # recommended preprocessing
        if is_normalization:
            std = StandardScaler()
            std.fit(X_train)
            X_train = std.transform(X_train)
            X_test = std.transform(X_test)

        img_file = f'out/{data_name}-correlation.png'
        df = pd.DataFrame(X_train, columns=columns)
        correlation(df, img_file, title=data_name)
        corr_feat_label(df['previous'], y_train)

        # Reduce the redundant features from X_train by VIF
        selected_columns = feature_selection_vif(X_train, vif_threshold=10)
        X_train = X_train[:, selected_columns]
        img_file = f'out/{data_name}-correlation-vif.png'
        df = pd.DataFrame(X_train, columns=[columns[j] for j in selected_columns])
        correlation(df, img_file, title=data_name)
