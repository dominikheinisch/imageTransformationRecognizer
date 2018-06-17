import numpy as np
from PIL import Image
from scipy.spatial import distance


class EuclideanDistanceHeuristic:
    def __init__(self, img_paths, lower_limit=0.01, upper_limit=0.3):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.default_number_to_fill = 0
        self.init_limits(img_paths)

    def init_limits(self, img_paths):
        img_1 = Image.open(img_paths[0])
        img_2 = Image.open(img_paths[1])
        size = max(sum(img_1.size), sum(img_2.size)) / 2
        self.lower_limit *= size
        self.upper_limit *= size
        self.default_number_to_fill = (self.lower_limit + self.upper_limit) / 2

    def are_pairs_correct(self, chosen_pairs):
        v0 = np.array([p[0].coords for p in chosen_pairs])
        v1 = np.array([p[1].coords for p in chosen_pairs])
        return self.check_pair(v0) and self.check_pair(v1)

    def check_pair(self, v):
        dist = distance.cdist(v, v, metric='euclidean')
        np.fill_diagonal(dist, self.default_number_to_fill)
        min_ = np.min(dist)
        max_ = np.max(dist)
        return self.lower_limit <= min_ and max_ <= self.upper_limit


class ShapeHeuristic:
    def __init__(self, upper_limit=0.3):
        self.upper_limit = upper_limit

    def are_pairs_correct(self, chosen_pairs):
        v0 = np.array([p[0].coords for p in chosen_pairs])
        v1 = np.array([p[1].coords for p in chosen_pairs])
        return self.check_pair(v0) and self.check_pair(v1)

    def check_pair(self, v):
        dist = distance.cdist(v, v, metric='euclidean')
        np.fill_diagonal(dist, dist[0, 1])
        return np.min(dist) / np.max(dist) > self.upper_limit
