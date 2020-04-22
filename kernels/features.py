"""
    File containing all the kernel computation
"""

import config
from os import mkdir, remove
from os.path import isdir, isfile, join
from multiprocessing import Pool
from utils.utils import print_verbose
from kernels.data import get_transformation
import numpy as np
from scipy.stats import binom
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances, KERNEL_PARAMS,euclidean_distances,manhattan_distances
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import KernelCenterer
from sklearn.utils.extmath import safe_sparse_dot
from kernels.kernels import *

def select_parameters(kernellist, data, verbose, force, n_jobs):
    """
        Heuristic search for parameters values
        By limiting the number of 0 and 1 values once kernel min max normalized
    
        Arguments:
            kernellist {List of name} -- Kernels parameters
            data {Array d*f} -- Data
    
        Returns:
            List of kernels and associated parameters
    """
    parameters = {}
    # Estimates the different parameters for each kernels

    #precompute necessary estimates
    if data.shape[0] > 200:
        data = data[np.random.choice(data.shape[0], 200, replace=False), :]

    eucldist = euclidean_distances(data, data, squared=True)
    mandist = manhattan_distances(data, data)
    inner = np.abs(safe_sparse_dot(data, data.T))

    # TODO : Better approximation of the parameters for ANOVA
    for kernel in kernellist:
        parameters_list = KERNEL_PARAMS[kernel]
        if len(parameters_list)  == 0:
            parameters[kernel] = None
        else:
            parameters[kernel] = {}
            if "degree" in parameters_list:
                parameters[kernel]["degree"] = [2,3]
            if "coef0" in parameters_list:
                parameters[kernel]["coef0"] = [0.1,1]
            if "gamma" in parameters_list:
                if kernel == "spherical":
                    gamma2 = np.median(eucldist)**2
                    parameters[kernel]["gamma"] = np.array([0.1,0.5,1.0])*gamma2   
                elif kernel == "polynomial":
                    gamma2 = 1.0/(np.median(inner))
                    parameters[kernel]["gamma"] = np.array([0.1,0.5,1.0,1.5])*gamma2
                elif kernel == "sigmoid":
                    gamma2 = 1.0/(np.median(inner))
                    parameters[kernel]["gamma"] = np.array([0.001,0.01])*gamma2
                elif kernel == 'laplacian':
                    gamma2 = 1.0/(np.median(mandist))
                    parameters[kernel]["gamma"] = np.array([0.2,0.5,1.0,1.5,2.0])*gamma2
                else:
                    gamma2 = 1.0/(np.median(eucldist))
                    parameters[kernel]["gamma"] = np.array([0.2,0.5,1.0,1.5,2.0])*gamma2                    
            if "c" in parameters_list:
                parameters[kernel]["c"] = np.array([0.1, 0.5, 1.])*np.median(eucldist)

    # Change format of parameters list
    parameters_list = []
    for kernel in parameters:
        if parameters[kernel] is None:
            parameters_list.append((kernel, None))
        else:
            paramset = [{}]
            for param, value in parameters[kernel].items():
                paramset = [{**ps, param:v} for ps in paramset for v in value]
            parameters_list.append((kernel, paramset))

    return parameters_list

def produce_kernels(dname, kernellist, data, verbose = 0, force = False, 
    n_jobs = 1, save_path = config.kerneldir,
    approximation = False, n_components = 150):
    """
        Computes the different kernels on the data and save them
        Autoselection of the hyperparameters using Median heuristic
        
        Arguments:
            kernellist {List of name} -- Kernels parameters
            data {Array d*f} -- Data
    
        Returns:
            List of name (len k), List of kernel: d*d array (len k)
    """
    if not(isdir(join(save_path, dname))):
        mkdir(join(save_path, dname))

    # Compute or open all the different kernels
    names, kernels = [], []
    for name, compute_data in get_transformation(data):
        names_transformation, kernels_transformation = [], []
        print_verbose("Computing kernels on {}".format(name), verbose)
        
        # Open existing data
        kernels_name = []
        if approximation:
            path = join(save_path, dname, name + '_approximated.npz')
        else:
            path = join(save_path, dname, name + '.npz')
        if isfile(path) and not(force):
            try:
                kernels_name = np.load(path)
                names_transformation = [k for k in kernels_name if any(kname in k for kname in kernellist)]
                kernels_transformation = [kernels_name[k] for k in names_transformation]
                print_verbose("File {} used".format(path), verbose)
            except:
                print_verbose("File at {} cannot be opened".format(path), verbose)
                remove(path)
        
        # Kernels already computed
        if len(names_transformation) > 0:
            names.extend(names_transformation)
            kernels.extend(kernels_transformation)
            continue

        # Estimate parameters for the given list of data
        kernelsParams = select_parameters(kernellist, compute_data, verbose, force, n_jobs)

        # Computes kernels
        for kernel, paramset in kernelsParams:
            if paramset:
                for params in paramset:
                    pstr = ' '.join(['{} {}'.format(key, value) for key, value in params.items() if value is not None])
                    kname = "{} - {}".format(name + kernel, pstr)
                    names_transformation.append(kname)

                    if approximation:
                        feature_map_nystroem = Nystroem(kernel = kernel, n_components = n_components, **params)
                        kernels_transformation.append(feature_map_nystroem.fit_transform(compute_data))
                    else:
                        kernels_transformation.append(pairwise_kernels(compute_data, metric=kernel, n_jobs=n_jobs, **params))
            else:
                names_transformation.append(name + kernel)
                if approximation:
                    feature_map_nystroem = Nystroem(kernel = kernel, n_components = n_components)
                    kernels_transformation.append(feature_map_nystroem.fit_transform(compute_data))
                else:
                    kernels_transformation.append(pairwise_kernels(compute_data, metric=kernel, n_jobs=n_jobs))

        try:
            np.savez_compressed(path, **{n: k for n, k in zip(names_transformation, kernels_transformation)})
        except:
            print("Kernels cannot be saved at {}".format(path), verbose)
            remove(path)
        names.extend(names_transformation)
        kernels.extend(kernels_transformation)

    return names, kernels

def normalize_and_check_kernels(names, kernels, number_cluster, normalize,
    check_method = None, clip = False, verbose = 0, n_jobs = 1):
    """
        Normalize the given kernels and verify their positiveness
        
        Arguments:
            names {List of str} -- List of names of the kernels
            kernels {List of array} -- List of all computed kernels
            number_cluster {int} -- Number cluster to compute
            normalize {str} -- Method to use for normalization
        
        Keyword Arguments:
            check_pos_def {bool} -- Verify the positiveness of kernels (default: {True})
        
        Returns:
            names, kernels -- List of names and normalized kernels
    """
    print_verbose("Normalization method : {}".format(normalize), verbose)

    if check_method is not None:
        if check_method == "pos_def":
            check_method = is_pos_def
        elif check_method == "trivial":
            check_method = is_trivial
        else:
            raise ValueError("Method for verification kernel : {} unknown".format(check_method))
        print_verbose("Verification kernel method : {}".format(check_method), verbose)

    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            normalized_kernels = pool.starmap(normalize_kernel, [(K, number_cluster, normalize, clip) for K in kernels])
    else:
        normalized_kernels = [normalize_kernel(K, number_cluster, normalize, clip) for K in kernels]

    norm_names, norm_kernels = [], []
    for name, ker in zip(names, normalized_kernels):
        if check_method is not None:
            try:
                test = check_method(ker)
            except:
                test = False
        elif ker is None:
            test = False
        else:
            test = True
            
        if test:
            print_verbose("Added: {}".format(name), verbose, level = 1)
            norm_names.append(name)
            norm_kernels.append(ker)
        else:
            print_verbose("NOT added: {}".format(name), verbose, level = 1)

    return norm_names, norm_kernels

def normalize_kernel(kernel, number_cluster, method, clip = False):
    """
        Normalize the kernel to make it comparable to the other
    
        Arguments:
            kernel {np array} -- Kernel matrix (n * n)
        
        Keyword Arguments:
            method {string} -- Method to use for normalization
            
        Returns:
            Array of indices of the kernel center
    """
    if method is None:
        normalized_kernel = kernel
    elif method == "spherical":
        normalized_kernel = normalize_kernel_spherical(kernel)
    elif method == "multiplicative":
        normalized_kernel = normalize_kernel_multiplicative(kernel)
    elif method == "multiplicative_center":
        normalized_kernel = normalize_kernel_center_multiplicative(kernel)        
    elif method == "expectation":
        normalized_kernel = normalize_kernel_expectation(kernel, number_cluster)
    elif method == "approximation":
        normalized_kernel = normalize_kernel_approximation(kernel)
    else:
        raise ValueError("Method for normalization kernel : {} unknown".format(method))

    if clip: 
        min_norm = np.nanmin(normalized_kernel[np.isfinite(normalized_kernel)]), 
        max_norm = np.nanmax(normalized_kernel[np.isfinite(normalized_kernel)])
        np.clip(normalized_kernel, min_norm, max_norm, normalized_kernel) # In place

    return normalized_kernel

def normalize_kernel_spherical(kernel):
    """
        Normalize a kernel in order to have a diagonal 1
        
        Arguments:
            kernel {Array n * n} -- Kernel

        Returns:
            Normalized kernel
    """
    d = kernel.diagonal()
    return kernel/np.sqrt(np.outer(d,d))

def normalize_kernel_multiplicative(kernel):
    """
        Normalize a kernel to have uniform variance (Ong and Zien (2008))

        Arguments:
            kernel {Array n * n} -- Kernel

        Returns:
            Normalized kernel
    """
    N = kernel.shape[0]
    return kernel / (np.trace(kernel)/N - np.sum(kernel)/(N**2))

def normalize_kernel_center_multiplicative(kernel):
    """
        Center, then normalize kernel to have uniform variance (Ong and Zien (2008))

        Arguments:
            kernel {Array n * n} -- Kernel

        Returns:
            Normalized kernel
    """
    N = kernel.shape[0]
    Kc = KernelCenterer().fit_transform(kernel)
    return Kc/(np.trace(Kc)/N)

def normalize_kernel_expectation(kernel, number_cluster):
    """
        Normalize a kernel with expected value of random clusters

        Arguments:
            kernel {Array n * n} -- Kernel
            number_cluster {int} -- Number of cluster

        Returns:
            Normalized kernel
    """
    N = kernel.shape[0]
    binom_law = binom(N-1, 1./number_cluster)
    denom = kernel.trace() * (1 - np.sum([binom_law.pmf(z)/(z+1) for z in np.arange(N)]))
    binom_law = binom(N-2, 1./number_cluster)
    denom -= (kernel.sum() - kernel.trace()) * np.sum([binom_law.pmf(z)/(z+2) for z in np.arange(N-1)]) / number_cluster
    return kernel / denom

def normalize_kernel_approximation(kernel, points = 500):
    """
        Normalizes an approximated kernel (results from Nystroem)

        Arguments:
            kernel {n * d Array} -- Nystroem approximation
            points {int} -- Number of points to use to approximate

        Returns:
            Normalized kernel
    """
    if len(kernel) > points:
        approx = kernel[np.random.choice(len(kernel), points, replace=False), :]
    else:
        approx = kernel

    inner = np.abs(safe_sparse_dot(approx, approx.T))
    denom = np.percentile(inner.flatten(), 95) / 2.0
    if denom < 0.0001:
        return None

    return kernel / denom

def is_pos_def(x):
    """
        Verifies if x is semi definite positive
        
        Arguments:
            x {Array n*n} -- Matrix
        
        Returns:
            Bool -- True if semi positive definite
    """
    return np.all(np.linalg.eigvals(x) >= 0)

def is_trivial(x, threshold = 0.6):
    """
        Verifies if a kernel is non trivial
        Check if contains any nan values
        and if mean value ratio above a given threshold
        
        Arguments:
            x {Array n*n} -- NORMALIZED Kernel

        Keyword Arguments:
            threshold {float} -- Ratio threshold (between 0 and 1)

        Returns:
            Bool -- True if non trivial
    """
    return np.all(~np.isnan(x)) and np.all(x.diagonal() >= 0) and (np.diagonal(x).mean() > 0) and (x.mean() / np.diagonal(x).mean() < threshold)