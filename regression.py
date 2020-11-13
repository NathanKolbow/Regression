# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:20:53 2020

@author: Nathan
"""

import pandas as pd
import numpy as np
from scipy.stats import t
import random


class RegressionModel():
    def __init__(self):
        self._data = None
        self._lm = None

    def read_csv(self, filename, *args, **kwargs):
        """
        INPUT: 
            filename - a string representing the path to the csv file
            *args    - unnamed arguments to be passed to Pandas' read_csv function
            **kwargs - named arguments to be passed to Pandas' read_csv function

        RETURNS:
            An n by m+1 array, where n is # data points and m is # features.
            The labels y should be in the first column.
        """
        self._data = pd.read_csv(filename, *args, **kwargs)

    def get_names(self):
        print(self._data.columns)

    def lm(self, formula):
        """
        INPUT:
            formula - the formula to fit the model on; currently the only valid
                      formula take the form "'A' ~ 'B' + 'C' + ..."

        RETURNS:
            An LMModel
        """
        vars = []
        i = 0
        while i < len(formula):
            # Find the next variable
            while formula[i] != '\'':
                i += 1

            j = i + 1
            while formula[j] != '\'':
                j += 1
            
            vars.append(formula[i+1:j]) 
            i = j + 1

        y = self._data.loc[:, vars[0]].values
        X = np.append(np.expand_dims(np.repeat(1, self._data.shape[0]), axis=1), self._data.loc[:, vars[1:]].values, axis=1)

        self._lm = LinearModel()
        self._lm.fit(X, y)

    def print_summary(self, model_type='lm'):
        if model_type == 'lm':
            if self._lm is not None:
                self._lm.print_summary()


        
class LinearModel():
    def __int__(self):
        self._betas = None

    def fit(self, X, y):
        self._n = X.shape[0]
        self._m = X.shape[1]

        self._betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        mse = 0
        for sample, actual in zip(X, y):
            mse += np.power((np.sum(self._betas * sample) - actual), 2)
        mse /= self._n - self._m

        self._sig_hat_sq = mse

        e = X.dot(self._betas) - y
        sig_sq = e.T.dot(e) / (self._n - self._m)
        self._se = np.sqrt(np.diag(sig_sq * np.linalg.inv(X.T.dot(X))))

        self._t = self._betas / self._se
        self._p = (1 - t.cdf(abs(self._t), self._n - self._m)) * 2

    def summary(self):
        if self._betas is None:
            return None

        return {
                    'n'      : self._n,
                    'm'      : self._m,
                    'betas'  : self._betas,
                    'se'     : self._se,
                    't'      : self._t,
                    'p-value': self._p
               }

    def print_summary(self):
        if self._betas is None:
            print("Not fitted model.")
            return

        summ = self.summary()
        print('\t\tEstimate\tStd.Error\tt value\tPr(>|t|)')
        for i in range(summ['m']):
            if i == 0:
                print('(Intercept)\t', end='')
            else:
                print('No name\t\t', end='')

            p = summ['p-value'][i]
            stars = ' ' if p > 0.1 else '.' if p > 0.05 else '*' if p > 0.01 else '**' if p > 0.001 else '***'
            print('%.3e\t%.3e\t%.3f\t%.4f %s' % (summ['betas'][i], summ['se'][i], summ['t'][i], p, stars))



if __name__ == '__main__':
    r = RegressionModel()
    r.read_csv("bodyfat.csv")
    r.lm("'BODYFAT' ~ 'DENSITY' + 'AGE' + 'KNEE'")
    r.print_summary()
    

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













