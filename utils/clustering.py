"""
    File containing embedding of clustering 
    Allows to use kmeans or DBSCAN
"""
from sklearn.cluster import KMeans, DBSCAN
from KernelConstrainedKmeans.initialization import Initialization, InitializationScale
from KernelConstrainedKmeans.wkckmeans import weightedKernelConstrainedKmeans

class Clustering:
    """
        Wrapper for clustering 
        This object is abstract and should be implemented
    """
    
    @classmethod
    def create(cls, method, **args):
        """
            Optimizer factory
        """
        if method == "kmeans":
            return Kmeans(**args)
        elif method == "kernelKmeans":
            return KernelKmeans(**args)
        elif method == "dbscan":
            return Dbscan(**args)
        else:
            print("Clustering algorithm unknown")

class Kmeans(Clustering):

    def __init__(self, classes, constraint_matrix, **args):
        self.initializer = InitializationScale(classes, constraint_matrix)
        self.classes = classes
        self.args = args

    def fit_transform(self, data, constraint_matrix = None):
        farthest_init = self.initializer.farthest_initialization(data) 
        if farthest_init is None:
            farthest_init = 'k-means++'
            n_init = 10
        else:
            n_init = 1
        return KMeans(n_clusters = self.classes, init = farthest_init, n_init = n_init, algorithm = 'elkan', **self.args).fit(data).labels_

class KernelKmeans(Clustering):

    def __init__(self, classes, constraint_matrix):
        self.initializer = Initialization(classes, constraint_matrix)
        self.classes = classes

    def fit_transform(self, data, constraint_matrix = None):
        farthest_init = self.initializer.farthest_initialization(data) 
        return weightedKernelConstrainedKmeans(data, farthest_init, max_iteration = 300, constraints=constraint_matrix)

class Dbscan(Clustering):

    def __init__(self, classes, constraint_matrix, **args):
        self.args = args

    def fit_transform(self, data, constraint_matrix = None):
        return DBSCAN(**self.args).fit(data).labels_