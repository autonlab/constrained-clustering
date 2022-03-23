# Constraint Satisfaction Clustering
This repository contains the code for comparing constrained clustering algorithms and reproduces the result from [Constrained Clustering and Multiple Kernel Learning without Pairwise Constraint Relaxation](). 

If you use our code, please cite

> [Constrained Clustering and Multiple Kernel Learning without Pairwise Constraint Relaxation]()\
Benedikt Boecking, Vincent Jeanselme, and Artur Dubrawski. arXiv preprint, 2022


## How to run our model on your dataset ?

`Example - MNIST.ipynb` provides a complete example on how to use the contrained clustering algorithms on the MNIST dataset.

### Constraint Satisfaction Clustering
#### Loading data and constraints
Constraints are represented as a sparse matrix, with +1 for must-link constraints and -1 for cannot-link constraints, and 0 otherwise. 
```python
import pandas as pd
import numpy as np

dname = 'data' # Dataset name - Used for saving kernels
ncluster = 3 # Number of clusters
data = pd.read_csv(dname).values
# see 'Example - MNIST.ipynb' for a more detailed example of how constraint matrices should be created
constraint = np.full((len(data), len(data)), 1) # Constraint matrix : +1 if linked, -1 otherwise - Prefer coomatrix
```

#### Create kernels for Kernel Constraint Satisfaction Clustering (KernelCSC)
```python
from models.kernel_opt import kernel_clustering
from kernels.features import produce_kernels, normalize_and_check_kernels

kernels_name = ['rbf', 'sigmoid', 'polynomial', 
                'laplacian', 'linear']
kernel_args = {"normalize": "expectation", 
               "check_method": "trivial", 
               "clip": True}

names, kernels = produce_kernels(dname, kernels_name, data, n_jobs = n_jobs)
names, kernels = normalize_and_check_kernels(names, kernels, ncluster, n_jobs = n_jobs, **kernel_args)
```

#### Run KernelCSC
```python
from utils.clustering import Clustering
clustering = Clustering.create("kernelKmeans", classes = ncluster, constraint_matrix = constraint)
assignment = kernel_clustering(kernels, clustering, constraint)
```

### Learn a Mahalanobis distance instead (MahalanobisCSC)
```python
from models.mahalanobis import mahalanobis_bayes_clustering
assignment = mahalanobis_bayes_clustering(data, clustering, constraint)
```

### Not sure about the number of clusters?
Use DBSCAN instead
```python
from utils.clustering import Clustering
clustering = Clustering.create("dbscan", classes = ncluster, constraint_matrix = constraint)
assignment = kernel_clustering(kernels, clustering, constraint)
```

### Large datasets but want to use KernelCSC?
Approximate the kernels 
```python
kernel_args = {"normalize": "approximation"}

names, kernels = produce_kernels(dname, kernels_name, data, n_jobs = n_jobs, approximation = True) # DO NOT FORGET THIS OPTION
names, kernels = normalize_and_check_kernels(names, kernels, ncluster, n_jobs = n_jobs, **kernel_args)
```
And executing the approximation version of the algorithm
```python
assignment = kernel_clustering(kernels, clustering, constraint, kernel_approx = True) # DO NOT FORGET THIS OPTION
```

## Dependencies
This code has been executed with `python3` with the dependencies indicated in `requirements.txt`. Additionaly, it is necessary to have R installed with the library `conclust`.
For setting up the environment:
```shell
conda create --name clustering
conda install -f -y -q --name clustering -c conda-forge --file requirements.txt
conda activate clustering
pip install pmlb==0.3 metric-learn==0.4.0 hyperopt
```

## Repository structure

### `data_files` and `kernel_files`
Any computed kernels and downloaded data will be saved in these directories

### `kernels`
This folder contains all functions relevant to the computation and normalization of kernels

### `models`
Contains all the proposed approaches to constrained clustering.

### `utils`
Contains additional functions for constraint completion, evaluation and optimization.
