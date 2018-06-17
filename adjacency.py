import numpy as np
from scipy.spatial import distance


def find_pairs_euclidean(l1, l2):
    v1 = np.array([x.vector for x in l1])
    v2 = np.array([x.vector for x in l2])
    distances = distance.cdist(v1, v2, metric='euclidean')
    pairs = []
    for i in range(len(l1)):
        corresponding = np.argmin(distances[i])
        if i == np.argmin(distances[:, corresponding]):
            pairs.append((l1[i], l2[corresponding]))
    return pairs


def filter_pairs(pairs, n, threshold):
    points1 = np.array([x[0] for x in pairs])
    points2 = np.array([x[1] for x in pairs])
    dist1 = calc_dist(points1)
    dist2 = calc_dist(points2)
    neighbors1 = get_n_smallest(dist1, n)
    neighbors2 = get_n_smallest(dist2, n)
    consistency = get_equivalent_size(neighbors1, neighbors2)
    indexes = selection(consistency, threshold)
    return np.array(pairs)[indexes]


def calc_dist(points):
    v = np.array([p.coords for p in points])
    return distance.cdist(v, v, metric='euclidean')


def get_n_smallest(dist, n):
    temp = np.argsort(dist, axis=1)[:, 0:n + 1]
    col_size = dist.shape[0]
    result = temp[:, 0:n]
    for i in range(col_size):
        for j in range(n):
            if result[i, j] == i:
                result[i, j] = temp[i, n]
    return result


def get_equivalent_size(neighbors1, neighbors2):
    col_size = neighbors1.shape[0]
    result = np.zeros(col_size)
    for i in range(col_size):
        result[i] = np.intersect1d(neighbors1[i], neighbors2[i]).shape[0]
    result /= neighbors1.shape[1]
    return result


def selection(consistency, threshold):
    # noinspection PyTypeChecker
    return np.argwhere(consistency > threshold).flatten()
