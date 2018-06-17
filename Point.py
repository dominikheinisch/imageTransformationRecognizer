class Point:
    def __init__(self, coords):
        self.coords = coords

    def __str__(self):
        return str(self.coords)

    def __repr__(self):
        return str(self.coords)


class FeaturedPoint(Point):
    def __init__(self, coords, vector):
        Point.__init__(self, coords)
        self.vector = vector
