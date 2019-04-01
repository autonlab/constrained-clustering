"""
    File containing functions linked to the constraint matrix
"""
import numpy as np
from tqdm import tqdm

def completion_constraint(constraint_matrix, force = False):
    """
        Complete the constraints matrix by
        forcing consistency and transitive closure
    
        Arguments:
            constraint_matrix {sparse array} -- Constrained on data points
                +1 => Constraint the points to be in the same cluster
                -1 => Constraint the points to be in separate clusters

        Returns:
            Completed constraint matrix {sparse array}
    """
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
    
    return result

def random_constraint(number_points):
    """
        Generates a random matrix of constraint
        
        Arguments:
            number_points {Int} -- Number of points

        Returns:
            Array number_points*number_points of -1, 0, 1
    """
    labelvector = np.random.randint(2, size=number_points)
    constraint = - np.ones((number_points, number_points))
    constraint += 2 * np.equal.outer(labelvector, labelvector)
    np.fill_diagonal(constraint, 0.)
    return constraint

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
    indices = np.array([(pi,pj) for i, pi in enumerate(list_points) for pj in list_points[:i]])
    return indices[np.random.choice(len(indices), size = number_indices, replace = False)]

def get_subselection(constraint_matrix, indices):
    """
        Returns the selection of the constraint_ matrix
        All other values are null
        
        Let also diagonal and symmetry

        Arguments:
            constraint_matrix {Array n*n} -- Matrix of constraints
            indices {List of (i int,j int)} -- Indices to keep 
    """
    selection = np.zeros_like(constraint_matrix)
    selection[indices[:,0], indices[:,1]] = 1
    selection[indices[:,1], indices[:,0]] = 1

    constraint_result = constraint_matrix.copy()
    constraint_result[np.logical_not(selection)] = 0
    return constraint_result

def verification_constraint(constraint_matrix, assignation):
    """
        Returns the number of constraint verified and broken
        
        Arguments:
            constraint_matrix {Array n*n} -- Constraint matrix
            assignation {Array n} -- Assignation

        Returns:
            number constraint respected, number constraint broken
    """
    observed_constraint = 2*np.equal.outer(assignation, assignation) - 1 # -1 different cluster, 1 same cluster
    comparison_assignation = np.multiply(observed_constraint, constraint_matrix) # 1 verified constraint, -1 unverified
    return np.sum(comparison_assignation > 0), np.sum(comparison_assignation < 0)
