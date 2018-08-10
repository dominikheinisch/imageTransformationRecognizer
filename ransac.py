import numpy as np
from scipy.spatial import distance
import itertools


class Ransac:
    def __init__(self, filtered_pairs):
        self._filtered_pairs = filtered_pairs
        self._model = None
        self._ransac_pairs = []

    def calculate(self, size, no_draws, max_error, heuristic=None):
        self.ransac_model(size, no_draws, max_error, heuristic)
        self.calculate_ransac_pairs(max_error)

    def ransac_model(self, size, no_draws, max_error, heuristic):
        pairs = self._filtered_pairs
        best_model = None
        best_score = 0
        for i in range(no_draws):
            model = None
            while model is None:
                indices = np.random.choice(pairs.shape[0], size=size)
                chosen = pairs[indices]
                if heuristic is None or heuristic.are_pairs_correct(chosen):
                    model = calc_model(chosen)
                else:
                    model = None
            score = 0
            for pair in pairs:
                score += 1 if model_error(model, pair) < max_error else 0
            if score > best_score:
                best_score = score
                best_model = model
        self._model = best_model

    def calculate_ransac_pairs(self, max_error):
        self._ransac_pairs = []
        for pair in self._filtered_pairs:
            if model_error(self._model, pair) < max_error:
                self._ransac_pairs.append(pair)

    def get_ransac_pairs(self):
        return self._ransac_pairs


def model_error(model, pair):
    return distance.cdist(np.reshape(model @ np.array([pair[0].coords[0], pair[0].coords[1], 1]), newshape=(1, -1)),
                          np.array([pair[1].coords[0], pair[1].coords[1], 1]).reshape(1, -1),
                          metric='euclidean')

def calc_model(samples):
    point_pairs = [(sample[0], sample[1]) for sample in samples]
    if len(samples) == 3:
        a = affine_array(point_pairs)
    elif len(samples) == 4:
        a = perspective_array(point_pairs)
    else:
        raise IndexError("incorrect samples size, allowed size is 3 or 4")
    return a


def affine_array(point_pairs):
    x1, y1, u1, v1, x2, y2, u2, v2, x3, y3, u3, v3 = get_params(point_pairs)
    a1 = np.array([[x1, y1, 1, 0, 0, 0],
                   [x2, y2, 1, 0, 0, 0],
                   [x3, y3, 1, 0, 0, 0],
                   [0, 0, 0, x1, y1, 1],
                   [0, 0, 0, x2, y2, 1],
                   [0, 0, 0, x3, y3, 1]])
    a2 = np.array([u1, u2, u3, v1, v2, v3])
    if not is_invertible(a1):
        return None
    a = np.linalg.inv(a1) @ a2
    res = np.reshape(np.append(a, [0, 0, 1]), newshape=(3, 3))
    return res


def perspective_array(point_pairs):
    x1, y1, u1, v1, x2, y2, u2, v2, x3, y3, u3, v3, x4, y4, u4, v4, = get_params(point_pairs)
    a1 = np.array([[x1, y1, 1, 0, 0, 0, -u1 * x1, -u1 * y1],
                   [x2, y2, 1, 0, 0, 0, -u2 * x2, -u2 * y2],
                   [x3, y3, 1, 0, 0, 0, -u3 * x3, -u3 * y3],
                   [x4, y4, 1, 0, 0, 0, -u4 * x4, -u4 * y4],
                   [0, 0, 0, x1, y1, 1, -v1 * x1, -v1 * y1],
                   [0, 0, 0, x2, y2, 1, -v2 * x2, -v2 * y2],
                   [0, 0, 0, x3, y3, 1, -v3 * x3, -v3 * y3],
                   [0, 0, 0, x4, y4, 1, -v4 * x4, -v4 * y4]])
    a2 = np.array([u1, u2, u3, u4, v1, v2, v3, v4])
    if not is_invertible(a1):
        return None
    a = np.linalg.inv(a1) @ a2
    # noinspection PyTypeChecker
    res = np.reshape(np.append(a, 1), newshape=(3, 3))
    return res


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def get_params(point_pairs):
    return tuple(itertools.chain.from_iterable([[pair[0].coords[0], pair[0].coords[1],
                                                 pair[1].coords[0], pair[1].coords[1]] for pair in point_pairs]))
