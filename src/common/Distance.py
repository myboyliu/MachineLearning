class Distance(object):
    @staticmethod
    def computeMinkowskiDistance(vector1, vector2, q ):
        distance = 0.
        n = len(vector1)
        for i in range(n):
            distance += pow(abs(float(vector1[i]) - float(vector2[i])), q)
        return round(pow(distance, 1.0 / q), 5)

    @staticmethod
    def computeManhattanDistance(vector1, vector2):
        return Distance.computeMinkowskiDistance(vector1, vector2, 1)

    @staticmethod
    def computeEuDistance(vector1, vector2):
        return Distance.computeMinkowskiDistance(vector1, vector2, 2)