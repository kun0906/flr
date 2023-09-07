import collections

import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_parts_info(parts):
    y2_dict = {}
    N2 = 0
    for X_, y_ in parts:
        for k, v in collections.Counter(y_).items():
            if k not in y2_dict.keys():
                N2 += v
                y2_dict[k] = v
            else:
                N2 += v
                y2_dict[k] += v

    return N2, y2_dict


def iid_partition(X, y, num_partitions, random_state):
    """ each client has the same distribution as the population, i.e., stratified sampling on y

    :param X:
    :param y:
    :param num_partitions:
    :param random_state:
    :return:
    """
    skf = StratifiedKFold(n_splits=num_partitions, random_state=random_state, shuffle=True)
    parts = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        parts.append((X[test_index], y[test_index]))
    # return parts

    # rng = np.random.RandomState(seed=random_state)
    # # Get unique class labels and their counts
    # unique_classes, class_counts = np.unique(y, return_counts=True)
    # parts = []
    # # Iterate through unique classes and create subsets
    # for index, (cnt, class_label) in enumerate(zip(class_counts, unique_classes)):
    #     class_indices = np.where(y == class_label)[0]
    #     rng.shuffle(class_indices)  # Shuffle indices
    #     X_, y_ = X[class_indices], y[class_indices]
    #     n_i = min(N_I, len(y_)//num_partitions)    # each client has almost equal number of points per class
    #     i = 0
    #     while i < num_partitions:
    #         st = i * n_i
    #         ed = (i + 1) * n_i
    #         if index == 0:
    #             parts.append((X_[st:ed], y_[st:ed]))
    #         else:
    #             tmp_X = np.concatenate([parts[i][0], X_[st:ed]], axis=0)
    #             tmp_y = np.concatenate([parts[i][1], y_[st:ed]])
    #             parts[i] = (tmp_X, tmp_y)
    #         i = i + 1
    #
    # # print(f'total rows: {sum([len(v[1]) for v in parts])}')
    N2, y2_dict = get_parts_info(parts)
    print(f'total rows: {N2}, {y2_dict.items()}')
    return parts


def noniid_partition(X, y, num_partitions, random_state):
    """ Non-IID data: each client have all the class data. each client has 100 points from each of rest classes.

    :param X:
    :param y:
    :param random_state:
    :return:
    """
    N = len(y)
    y_dict = collections.Counter(y)

    rng = np.random.RandomState(seed=random_state)
    # Get unique class labels and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    parts = []
    for i in range(num_partitions):
        tmp = ()
        for index, (cnt, class_label) in enumerate(zip(class_counts, unique_classes)):
            class_indices = np.where(y == class_label)[0]
            rng.shuffle(class_indices)  # Shuffle indices
            assert len(class_indices) != 100
            used_indexes = class_indices[:100]
            X_, y_ = X[used_indexes], y[used_indexes]
            mask = np.full(len(y), True)
            mask[used_indexes] = False
            X, y = X[mask], y[mask]
            if index == 0:
                tmp = (X_, y_)
            else:
                tmp_X = np.concatenate([tmp[0], X_], axis=0)
                tmp_y = np.concatenate([tmp[1], y_])
                tmp = (tmp_X, tmp_y)  # be careful of the index here
        parts.append(tmp)

    N2, y2_dict = get_parts_info(parts)
    print(f'total rows: {N2}, {y2_dict.items()}')

    # for the rest data
    unique_classes, class_counts = np.unique(y, return_counts=True)
    left_parts = num_partitions
    start_index = 0
    for index, (cnt, class_label) in enumerate(zip(class_counts, unique_classes)):
        class_indices = np.where(y == class_label)[0]
        rng.shuffle(class_indices)  # Shuffle indices
        used_indexes = class_indices
        X_, y_ = X[used_indexes], y[used_indexes]
        mask = np.full(len(y), True)
        mask[used_indexes] = False
        X, y = X[mask], y[mask]
        if index == len(unique_classes) - 1:
            n_parts = left_parts
        else:
            assert num_partitions >= len(unique_classes)
            n_parts = num_partitions // len(unique_classes)  # even partition the rest of partitions
            left_parts = left_parts - n_parts

        assert len(class_indices) >= n_parts
        n_i = len(class_indices) // n_parts
        for j in range(n_parts):
            if j == n_parts - 1:
                X2, y2 = X_[j * n_i:, :], y_[j * n_i:]
            else:
                X2, y2 = X_[j * n_i:(j + 1) * n_i, :], y_[j * n_i:(j + 1) * n_i]
            tmp = parts[start_index + j]
            tmp_X = np.concatenate([tmp[0], X2], axis=0)
            tmp_y = np.concatenate([tmp[1], y2])
            parts[start_index + j] = (tmp_X, tmp_y)
        if n_parts > 0: start_index += 1

    N2, y2_dict = get_parts_info(parts)
    print(f'total rows: {N}, {y_dict.items()}')
    print(f'total rows: {N2}, {y2_dict.items()}')
    assert N == N2
    assert y_dict == y2_dict

    return parts


def sample_iid_partition(X, y, num_partitions, N_I, random_state):
    """ Each client has N_I data points with the same class distribution as the population,
    i.e., stratified sampling on y


    :param X:
    :param y:
    :param num_partitions:
    :param N_I:
    :param random_state:
    :return:
    """
    skf = StratifiedKFold(n_splits=num_partitions, random_state=random_state, shuffle=True)
    parts = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_i, Y_i = X[test_index], y[test_index]
        # the test_index is for each client
        rng = np.random.RandomState(seed=random_state)
        # Get unique class labels and their counts
        unique_classes, class_counts = np.unique(Y_i, return_counts=True)
        # Iterate through unique classes and create subsets
        tmp = ()
        for index, (cnt, class_label) in enumerate(zip(class_counts, unique_classes)):
            class_indices = np.where(Y_i == class_label)[0]
            rng.shuffle(class_indices)  # Shuffle indices
            # weight for each class
            if index == len(unique_classes) - 1:
                n_i = max(N_I - len(tmp[1]), 1)
            else:
                n_i = max(int(np.floor(len(class_indices) / len(Y_i) * N_I)), 1)
            class_indices = class_indices[:n_i]
            # print(index, n_i)
            if index == 0:
                tmp = ((X_i[class_indices], Y_i[class_indices]))
            else:
                tmp_X = np.concatenate([tmp[0], X_i[class_indices]], axis=0)
                tmp_y = np.concatenate([tmp[1], Y_i[class_indices]])
                tmp = (tmp_X, tmp_y)
                # print(Y_i[class_indices])
        # print(collections.Counter(tmp[1]))
        parts.append(tmp)
    N2, y2_dict = get_parts_info(parts)
    print(f'total rows: {N2}, {y2_dict.items()}')
    return parts

    # if N_I <=0:
    #     N_I = len(y)
    #
    # rng = np.random.RandomState(seed=random_state)
    # # Get unique class labels and their counts
    # unique_classes, class_counts = np.unique(y, return_counts=True)
    # parts = []
    # # Iterate through unique classes and create subsets
    # for index, (cnt, class_label) in enumerate(zip(class_counts, unique_classes)):
    #     class_indices = np.where(y == class_label)[0]
    #     rng.shuffle(class_indices)  # Shuffle indices
    #     X_, y_ = X[class_indices], y[class_indices]
    #     n_i = min(N_I, len(y_) // num_partitions)  # each client has almost equal number of points per class
    #     i = 0
    #     while i < num_partitions:
    #         st = i * n_i
    #         ed = (i + 1) * n_i
    #         if index == 0:
    #             parts.append((X_[st:ed], y_[st:ed]))
    #         else:
    #             tmp_X = np.concatenate([parts[i][0], X_[st:ed]], axis=0)
    #             tmp_y = np.concatenate([parts[i][1], y_[st:ed]])
    #             parts[i] = (tmp_X, tmp_y)
    #         i = i + 1
    #
    # # print(f'total rows: {sum([len(v[1]) for v in parts])}')
    return parts


def sample_noniid_partition(X, y, num_partitions, N_I, random_state):
    """ Non-IID data: each client have (N_I + N_I * 0.1 * (C-1)) class data. each client has 10% data points
    from each of other classes

    :param X:
    :param y:
    :param random_state:
    :return:
    """
    N = len(y)
    y_dict = collections.Counter(y)

    rng = np.random.RandomState(seed=random_state)
    # Get unique class labels and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    parts = []
    for i in range(num_partitions):
        tmp = ()
        for index, (cnt, class_label) in enumerate(zip(class_counts, unique_classes)):
            class_indices = np.where(y == class_label)[0]
            rng.shuffle(class_indices)  # Shuffle indices
            n_i = int(np.floor(N_I * 0.1))  # each client has 10% point from other classes
            assert len(class_indices) != n_i
            used_indexes = class_indices[:n_i]
            X_, y_ = X[used_indexes], y[used_indexes]
            mask = np.full(len(y), True)
            mask[used_indexes] = False
            X, y = X[mask], y[mask]
            if index == 0:
                tmp = (X_, y_)
            else:
                tmp_X = np.concatenate([tmp[0], X_], axis=0)
                tmp_y = np.concatenate([tmp[1], y_])
                tmp = (tmp_X, tmp_y)  # be careful of the index here
        parts.append(tmp)

    N2, y2_dict = get_parts_info(parts)
    print(f'total rows: {N2}, {y2_dict.items()}')

    # for the rest data
    unique_classes, class_counts = np.unique(y, return_counts=True)
    left_parts = num_partitions
    start_index = 0
    for index, (cnt, class_label) in enumerate(zip(class_counts, unique_classes)):
        class_indices = np.where(y == class_label)[0]
        rng.shuffle(class_indices)  # Shuffle indices
        used_indexes = class_indices
        X_, y_ = X[used_indexes], y[used_indexes]
        mask = np.full(len(y), True)
        mask[used_indexes] = False
        X, y = X[mask], y[mask]     # updated X, y
        if index == len(unique_classes) - 1:
            n_parts = left_parts
        else:
            assert num_partitions >= len(unique_classes)
            n_parts = num_partitions // len(unique_classes)  # even partition the rest of partitions
            left_parts = left_parts - n_parts

        assert len(class_indices) >= n_parts
        n_i = max(N_I - int(np.floor(N_I * 0.1)),1)  # each client has N_I data for one class
        for j in range(n_parts):
            X2, y2 = X_[j * n_i:(j + 1) * n_i, :], y_[j * n_i:(j + 1) * n_i]
            tmp = parts[start_index + j]
            tmp_X = np.concatenate([tmp[0], X2], axis=0)
            tmp_y = np.concatenate([tmp[1], y2])
            parts[start_index + j] = (tmp_X, tmp_y)
            print(collections.Counter(tmp_y))
        if n_parts > 0: start_index += 1

    N2, y2_dict = get_parts_info(parts)
    print(f'total rows: {N}, {y_dict.items()}')
    print(f'total rows: {N2}, {y2_dict.items()}')
    # assert N == N2
    # assert y_dict == y2_dict

    return parts

#
# def sample_noniid_partition_old(X, y, num_partitions, N_I, random_state):
#     """ Non-IID data: each client have all the class data. each client has N_I point from a class and 10% other points
#     from the rest classes.
#     This scripts might include duplicates during to the sampling on th 10% data from other classes
#
#     :param X:
#     :param y:
#     :param N_I: each client should have n_i points for each class, if N_I == -1, we use all the data.
#     :param random_state:
#     :return:
#     """
#     if N_I <= 0:
#         N_I = len(y)
#
#     rng = np.random.RandomState(seed=random_state)
#     # Get unique class labels and their counts
#     unique_classes, class_counts = np.unique(y, return_counts=True)
#     parts = []
#     # Iterate through unique classes and create subsets
#     left_partitions = num_partitions
#     for index, (cnt, class_label) in enumerate(zip(class_counts, unique_classes)):
#         class_indices = np.where(y == class_label)[0]
#         rng.shuffle(class_indices)  # Shuffle indices
#         X_, y_ = X[class_indices], y[class_indices]
#         n_i = min(N_I,
#                   len(y_) // num_partitions)  # each client has almost equal number of points per class, if N_I == -1, we will use all the points
#         if index == len(unique_classes) - 1:  # for the last class, we use all the rest partitions
#             num_partitions_i = left_partitions
#         else:
#             num_partitions_i = num_partitions // len(unique_classes)  # each class has how many partitions.
#             left_partitions = left_partitions - num_partitions_i
#         start_index = len(parts)
#         i = 0
#         while i < num_partitions_i:
#             st = i * n_i
#             ed = (i + 1) * n_i
#             parts.append((X_[st:ed], y_[st:ed]))
#             # sampling 10% data from other classes
#             tmp_X = parts[-1][0]  # be careful of the index here
#             tmp_y = parts[-1][1]
#             for class_label2 in unique_classes:
#                 if class_label2 == class_label: continue
#                 class_indices2 = np.where(y == class_label2)[0]
#                 rng.shuffle(class_indices2)  # Shuffle indices
#                 n2_i = int(np.floor(n_i * 0.1))  # sampling 10% data from other classes
#                 class_indices2 = class_indices2[:n2_i]
#                 X2_, y2_ = X[class_indices2], y[class_indices2]
#                 tmp_X = np.concatenate([tmp_X, X2_], axis=0)
#                 tmp_y = np.concatenate([tmp_y, y2_])
#             parts[-1] = (tmp_X, tmp_y)  # be careful of the index here
#             i = i + 1
#
#     print(f'total rows: {sum([len(v[1]) for v in parts])}, {len(y)}')
#     return parts
