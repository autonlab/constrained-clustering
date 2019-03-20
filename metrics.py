"""
    File containing metrics to evaluate the quality of clustering
"""
import numpy as np
from sklearn import metrics

def evalCluster(predictions, labelGT):
    """
        Computes different metrics on the clustering reslts
        
        Arguments:
            predictions {Array} -- Results of the clustering algo
            labelGT {Array} -- Ground truth label
        
        Returns:
            Dictionary of metrics
    """

    cluster_metric = {}
    cluster_metric["Adjusted Rand"] = metrics.adjusted_rand_score(labelGT, predictions)  
    cluster_metric["Adjuster Mutual Info"] = metrics.adjusted_mutual_info_score(labelGT, predictions)  
    cluster_metric["Normalized Mutual Info"] = metrics.normalized_mutual_info_score(labelGT, predictions)  
    cluster_metric["Fowlkes Mallows"] = metrics.fowlkes_mallows_score(labelGT, predictions)
    cluster_metric["FScore"] = fScore(labelGT, predictions)
    return cluster_metric

def evalSplit(predictions, labelGT, trainIndices):
    """
        Computes the metrics on the different splits of the data
    
        Arguments:
            predictions {Array int} -- Results of the clustering algo
            labelGT {Array int} -- Ground truth label
            trainIndices {Array int} -- Indices of the training points
        
        Returns:
            Dictionary with perf on all, on test and train splits
    """

    testIndices = [i for i in np.arange(len(predictions)) if i not in trainIndices]
    return {"all":      evalCluster(predictions, labelGT), 
            "test":     evalCluster(predictions[testIndices], labelGT[testIndices]), 
            "train":    evalCluster(predictions[trainIndices], labelGT[trainIndices])}

def fScore(predictions, labelGT):
    """
        Computes the f score for the observed constraint
    
        Arguments:
            predictions {Array int} -- Results of the clustering algo
            labelGT {Array int} -- Ground truth label
        
        Returns:
            F score {float}
    """
    observed = 2 * np.equal.outer(predictions, predictions) - 1
    constraint = 2 * np.equal.outer(labelGT, labelGT) - 1
    selection = np.tril_indices_from(constraint, 1)
    selection = selection[np.array(constraint[selection] != 0)]
    return metrics.f1_score(constraint[selection], observed[selection])