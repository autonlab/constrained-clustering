# ConstrainedClustering
Repository for comparison of constraint clustering algorithms. 

## Structure
In order to reproduce the experiment execute the norebook `Experiments.ipynb` and then `Visualizations.ipynb` which allows to analyze the results and compare the different methods.  
All the other functions contains functions used for the clustering. 

## Dependencies
This code has been executed with `python3` with the library indicated in `requirements.txt`. Additionaly, it is necessary to have R installed with the library `conclust`.
For setting up the environment:
```
conda create --name clustering
conda install -f -y -q --name clustering -c conda-forge --file requirements.txt
conda activate clustering
pip install pmlb metric-learn==0.4.0 hyperopt
```

## How to compare your method ?
Import your library in `Experiments.ipynb` and select the algorithms that you want to run. The execution of the notebook saves all results in the result folder (indicated in `config.py` with the timestamp indicated when you run the experiments notebook). Copy this timestamp in the `dates` list in `Visualizations.ipynb` to compare the results (you can indicate several dates to compare methods computed at different time). 

## How to run our model on your dataset ?

### Kernel Clustering
#### Open your data and constraints
```
import pandas as pd
import numpy as np

dname = 'data' # Data name - Used for saving kernels
ncluster = 3 # Number cluster
data = pd.read_csv(dname)
constraint = np.full((len(data), len(data)), 1) # Constraint matrix : +1 if linked, -1 otherwise - Prefer coomatrix
```

#### Create kernels
```
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

#### Run Kmeans and opitmization
```
from utils.clustering import Clustering
clustering = Clustering.create("kmeans", classes = ncluster, constraint_matrix = constraint)
assignment = kernel_clustering(kernels, clustering, constraint)
```

### Use Mahalanobius 
```
from models.mahalanobis import mahalanobis_bayes_clustering

assignment = mahalanobis_bayes_clustering(data, clustering, constraint)
```

## Not sure about cluster number ?
Use DBSCAN instead
```
from utils.clustering import Clustering
clustering = Clustering.create("dbscan", classes = ncluster, constraint_matrix = constraint)
assignment = kernel_clustering(kernels, clustering, constraint)
```
