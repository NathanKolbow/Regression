# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:20:53 2020

@author: Nathan
"""

import pandas as pd
import numpy as np
import random


def get_dataset(filename):
    """
    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    df = pd.read_csv(filename)
    del df["IDNO"]
    
    return df.values
    

def print_stats(dataset, col):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    PRINTS:
        n              - number of samples
        sample mean    - sample mean of the given feature
        sample std dev - sample standard deviation of the given fetaure

    RETURNS:
        None
    """
    n = dataset.shape[0]
    samp_mean = np.sum(dataset[:, col]) / n
    samd_sd = np.sqrt(np.sum((dataset[:, col] - samp_mean) ** 2) / (n-1))

    print(n)
    print("%.2f" % samp_mean)
    print("%.2f" % samd_sd)
    

def regression(dataset, cols, betas):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    mse = 0
    for sample in dataset:
        prediction = betas[0] + np.sum(sample[cols] * np.array(betas[1:]))
        mse += ((prediction - sample[0]) ** 2)
    return mse / dataset.shape[0]


def gradient_descent(dataset, cols, betas):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        A 1D array of gradients
    """
    n = dataset.shape[0]
    gradient = np.zeros(len(betas))
    for sample in dataset:
        prediction = betas[0] + np.sum(sample[cols] * np.array(betas[1:]))
        gradient[0] += prediction - sample[0]
        gradient[1:] += (prediction - sample[0]) * sample[cols]
    
    return np.apply_along_axis(lambda x : 2 * x / n, 0, gradient)


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    PRINTS (for each iteration t in [1, T]):
        t     - current iteration
        mse   - mean squared error of the current model iteration
        betas - the betas associated with the current model iteration

    RETURNS:
        None
    """
    for t in range(T):
        betas -= eta * gradient_descent(dataset, cols, betas)
        mse = regression(dataset, cols, betas)
        
        output = f'{t+1} {mse:.2f} {betas[0]:.2f}'
        for i in range(1, len(betas)):
            output += f' {betas[i]:.2f}'
        print(output)


def compute_betas(dataset, cols):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    X = np.append(np.expand_dims(np.repeat(1, dataset.shape[0]), axis=1), dataset[:, cols], axis=1)
    y = dataset[:, 0]
    
    betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    mse = regression(dataset, cols, betas)
    
    return tuple(np.append(mse, betas))


def predict(dataset, cols, features):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = np.array(compute_betas(dataset, cols)[1:])
    return betas[0] + np.sum(features * betas[1:])


def random_index_generator(min_val, max_val, seed=42):
    """
    DO NOT MODIFY THIS FUNCTION.
    DO NOT CHANGE THE SEED.
    This generator picks a random value between min_val and max_val,
    seeded by 42.
    """
    random.seed(seed)
    while True:
        yield random.randrange(min_val, max_val)


def sgd(dataset, cols, betas, T, eta):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    PRINTS (for each iteration t in [1, T]):
        t     - current iteration
        mse   - mean squared error of the current model iteration
        betas - the betas associated with the current model iteration

    RETURNS:
        None
    """
    gen = random_index_generator(0, dataset.shape[0] - 1)
    for t in range(T):
        _sample = gen.__next__()
        betas -= eta * gradient_descent(dataset[_sample, :].reshape(1, -1), cols, betas)
        mse = regression(dataset, cols, betas)
        
        output = f'{t+1} {mse:.2f} {betas[0]:.2f}'
        for i in range(1, len(betas)):
            output += f' {betas[i]:.2f}'
        print(output)













