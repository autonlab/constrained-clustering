# ConstrainedClustering
Repository for comparison of constraint clustering algorithms. See [A Perspective on Clustering with Pairwise Constraints]() by Benedikt Boecking, Vincent Jeanselme and Artur Dubrawski (Carnegie Mellon University - Auton Lab - Robotics Institute)

## Structure
In order to reproduce the experiment execute the norebook `Experiments.ipynb` and then `Visualizations.ipynb` which allows to analyze the results and compare the different methods.  
All the other functions contains functions used for the clustering. 

## Dependencies
This code has been executed with `python3` with the library indicated in `requirements.txt`. Additionaly, it is necessary to have R installed with the library `conclust`.

## How to compare your method ?
Import your library in `Experiments.ipynb` and select the algorithms that you want to run. The execution of the notebook saves all results in the result folder (indicated in `config.py` with the timestamp indicated when you run the experiments notebook). Copy this timestamp in the `dates` list in `Visualizations.ipynb` to compare the results (you can indicate several dates to compare methods computed at different time). 
