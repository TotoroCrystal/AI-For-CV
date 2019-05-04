import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class KMeansClassifier():
    "This is a k-means classifier"

    def __init__(self, k = 3, initCent = 'random', max_iter = 500):
        self._k = k
        self._initCent = initCent
        self._max_iter = max_iter
        self._clusterAssment = None
        self._labels = None
        self._sse = None

    def _calEDist(self, arrA, arrB):
        """
        :param arrA: 1-d array
        :param arrB: 1-d array
        :return: Euler Distance
        """
        return np.math.sqrt(sum(np.power(arrA - arrB, 2)))

    def _calMDist(self, arrA, arrB):
        """
        :param arrA: 1-d array
        :param arrB: 1-d array
        :return: Manhattan Distance
        """
        return sum(np.abs(arrA - arrB))

    def _randCent(self, data_X, k):
        """
        :param data_X: feature points
        :param k: number of centroid
        :return: k centroids, m*n centroid matrix
        """
        n = data_X.shape[1]   # get features dimension
        centroids = np.empty((k, n))   # using numpy to generate a k*n matrix to store centroids
        for j in range(n):
            minJ = min(data_X[:, j])
            rangeJ = float(max(data_X[:, j] - minJ))
            # flatten nested list
            centroids[:, j] = (minJ + rangeJ * np.random.rand(k, 1)).flatten()
        return centroids

    def fit(self, data_X):
        """
        :param data_X: input m*n matrix
        :return:
        """
        if not isinstance(data_X, np.ndarray) or isinstance(data_X, np.matrixlib.defmatrix.matrix):
            try:
                data_X = np.asarray(data_X)
            except:
                raise TypeError("numpy.ndarray required for data_X")
        m = data_X.shape[0]   # number of sample
        # create a m*2 2-d matrix, first column stores cluster index of sample point
        # second column stores SSE
        self._clusterAssment = np.zeros((m, 2))

        if self._initCent == 'random':
            self._centroids = self._randCent(data_X, self._k)

        clusterChanged = True
        for _ in range(self._max_iter):
            clusterChanged = False
            for i in range(m):   # assign sample point to the cluster of nearliest centroid
                minDist = np.inf   # initializing minDist to infinite value
                minIndex = -1   # set nearliest centroid index to -1
                for j in range(self._k):  # this iteration aims to find nearliest centroid
                    arrA = self._centroids[j, :]
                    arrB = data_X[i, :]
                    distJI = self._calEDist(arrA, arrB)  # calculate euler distance
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j

                if self._clusterAssment[i, 0] != minIndex or self._clusterAssment[i, 1] > minDist**2:
                    clusterChanged = True
                    self._clusterAssment[i, :] = minIndex, minDist**2
            if not clusterChanged:   # if all clusters don't change, means convergence and stop iteration
                break
            for i in range(self._k):  # update centroids, set mean value of points of each cluster
                index_all = self._clusterAssment[:, 0]  # all index of samples in each cluster
                value = np.nonzero(index_all == i)   # index of i-th cluster
                ptsInClust = data_X[value[0]]   # all points of i-th cluster
                self._centroids[i, :] = np.mean(ptsInClust, axis = 0)   # calculate mean value

        self._labels = self._clusterAssment[:, 0]
        self._sse = sum(self._clusterAssment[:, 1])

    def predict(self, X):   # based on clusters results, predict new point belong to which luster
        # type chack
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]   # sample number
        preds = np.empty((m, ))
        for i in range(m):   # assign sample point to cluster of closest centroid
            minDist = np.inf
            for j in range(self._k):
                distJI = self._calEDist(self._centroids[j, :], X[i, :])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds


def loadDataset(infile):
    df = pd.read_csv(infile, sep = '\t', header = 0, dtype = str, na_filter = False)
    return np.array(df).astype(np.float)

if __name__ == "__main__":
    data_X = loadDataset(r"./testSet.txt")
    k = 5
    clf = KMeansClassifier(k)
    clf.fit(data_X)
    cents = clf._centroids
    labels = clf._labels
    sse = clf._sse
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845856']
    for i in range(k):
        index = np.nonzero(labels == i)[0]
        x0 = data_X[index, 0]
        x1 = data_X[index, 1]
        y_i = i
        for j in range(len(x0)):
            plt.text(x0[j], x1[j], str(y_i), color=colors[i], fontdict={'weight': 'bold', 'size': 6})
        plt.scatter(cents[i, 0], cents[i, 1], marker = 'x', color = colors[i], linewidths = 7)

    plt.title("SSE = {:.2f}".format(sse))
    plt.axis([-7, 7, -7, 7])
    plt.show()








