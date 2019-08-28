"""
    File containing functions linked to the constraint matrix
"""
import numpy as np
from numba import jit
from tqdm import tqdm
from scipy.sparse import coo_matrix, find

def completion_constraint(constraint_matrix, force = False):
    """
        Complete the constraints matrix by
        forcing consistency and transitive closure

        NB: Matrix will be dense
    
        Arguments:
            constraint_matrix {sparse array} -- Constrained on data points
                +1 => Constraint the points to be in the same cluster
                -1 => Constraint the points to be in separate clusters

        Returns:
            Completed constraint matrix {sparse array}
    """
    constraint_matrix = constraint_matrix.todense()
    assert np.array_equal(constraint_matrix.T, constraint_matrix)

    # Transitive closure on positive constraint
    # Adaptated Floyd–Warshall algorithm
    positive = np.where(constraint_matrix > 0, constraint_matrix, np.zeros_like(constraint_matrix))
    negative = np.where(constraint_matrix < 0, constraint_matrix, np.zeros_like(constraint_matrix))
    notNull = np.unique(np.argwhere(constraint_matrix != 0))
    for k in notNull:
        for end, i in enumerate(notNull):
            for j in notNull[:end]:
                # Improved version for going faster
                value = positive[i, k] * positive[k, j]
                if positive[i, j] < value:
                    positive[i, j] = value
                    positive[j, i] = value

                value = positive[i, k] * negative[k, j] + negative[i, k] * positive[k, j]
                if negative[i, j] > value:
                    negative[i, j] = value
                    negative[j, i] = value

    if not(force):
        # Verify that no opposite constraint
        assert np.sum(np.multiply(positive, constraint_matrix) < 0) == 0, "Transitive Closure breaks constraint (use force option to erase the less sure constraint)"
        assert np.sum(np.multiply(negative, constraint_matrix) < 0) == 0, "Transitive Closure breaks constraint (use force option to erase the less sure constraint)"

    # Take the most confident constraint
    result = np.where(positive >= np.abs(constraint_matrix), positive, constraint_matrix)
    result = np.where(np.abs(negative) >= np.abs(result), negative, result)
    
    return coo_matrix(result)

def random_constraint(number_points):
    """
        Generates a random matrix of constraint
        
        Arguments:
            number_points {Int} -- Number of points

        Returns:
            Array number_points*number_points of -1, 0, 1
    """
    labelvector = np.random.randint(2, size = number_points)
    return generate_constraint(labelvector, number_points * (number_points - 1) / 2)

def random_indices(list_points, number_indices):
    """
        Generates a list of indices to apply on the constraint matrix
        without redundancy
        
        Arguments:
            list_points {List of Int / Int} -- Number of points in dataset or list of points to take into account
            number_indices {Int} -- Number of indices needed

        Returns:
            List of pairs of coordinates
    """
    if isinstance(list_points, int):
        list_points = np.arange(list_points)

    length = len(list_points)
    indices = set()
    while len(indices) < number_indices:
        i = np.random.randint(length - 1)
        j = np.random.randint(i + 1, length)
        indices.add((list_points[i], list_points[j]))

    return list(indices)

def generate_constraint(labels, indices):
    """
        Returns the sparse matrix of constraints

        Arguments:
            labels {Array n} -- Ground truth labels
            indices {List of (i int, j int)} -- Indices to keep 
    """
    rows, cols, vals = [], [], []
    for i, j in indices:
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([1 if (labels[i] == labels[j]) else -1] * 2)

    return coo_matrix((vals, (rows, cols)), shape = (len(labels), len(labels)))

def verification_constraint(constraint_matrix, assignation):
    """
        Returns the number of constraint verified and broken
        
        Arguments:
            constraint_matrix {Array n*n} -- Constraint matrix
            assignation {Array n} -- Assignation

        Returns:
            number constraint respected, number constraint broken
    """
    @jit(nopython=True)
    def fast_verification(row, col, data, assingation):
        respected, broken = 0, 0
        for i, j, val in zip(row, col, data):   
            if assignation[i] == assignation[j] and val > 0:
                respected += 1
            elif assignation[i] != assignation[j] and val < 0:
                respected += 1
            else:
                broken += 1
        return respected, broken

    return fast_verification(constraint_matrix.row, constraint_matrix.col, constraint_matrix.data, assignation)

def indices_constraint(constraint_matrix, assignation):
    """
        Returns the number of constraint verified and broken
        
        Arguments:
            constraint_matrix {Array n*n} -- Constraint matrix
            assignation {Array n} -- Assignation

        Returns:
            number constraint respected, number constraint broken
    """
    @jit(nopython=True)
    def fast_indices(row, col, data, assingation):
        respected, broken = [], []
        for i, j, val in zip(row, col, data):   
            if assignation[i] == assignation[j] and val > 0:
                respected.append((i,j))
            elif assignation[i] != assignation[j] and val < 0:
                respected.append((i,j))
            else:
                broken.append((i,j))
        return respected, broken

    return fast_indices(constraint_matrix.row, constraint_matrix.col, constraint_matrix.data, assignation)

def kta_score(constraint_matrix, assignation):
    """
        Returns the kta score
        
        Arguments:
            constraint_matrix {Array n*n} -- Constraint matrix
            assignation {Array n} -- Assignation

        Returns:
            KTA Score
    """
    respected, broken = verification_constraint(constraint_matrix, assignation)
    return respected / (respected + broken)