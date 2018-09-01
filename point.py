import math

class Point:
    def __init__(self, coords):
        self.coords = coords

    def __eq__(self, other):
        zipped = zip(self.coords, other.coords)
        return len(self.coords) == len(other.coords) and \
               sum(list(map(lambda x: math.isclose(x[0], x[1], rel_tol=1e-6), zipped))) == len(self.coords)

    def __ne__(self, other):
        return not __eq__(other)


    def __str__(self):
        return str(self.coords)

    def __repr__(self):
        return str(self.coords)


class FeaturedPoint(Point):
    def __init__(self, coords, vector):
        Point.__init__(self, coords)
        self.vector = vector
