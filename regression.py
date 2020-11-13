import pandas as pd
import numpy as np
from scipy.stats import t
import random


class Regressor():
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
        y_var, x_vars = formula.split('~')
        x_vars = x_vars.split('+')
        y_var = y_var.strip()
        for i in range(len(x_vars)):
            x_vars[i] = x_vars[i].strip()

        y = self._data.loc[:, y_var]
        X = self._data.loc[:, x_vars]
        X['(Intercept)'] = 1

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

        self._names = np.append(y.name, X.columns)
        y = y.values
        X = X.values

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
        max_len = max([len(x) for x in self._names])
        for i in range(summ['m']):
            print(self._names[i+1].ljust(max_len), '\t', end='')

            p = summ['p-value'][i]
            stars = ' ' if p > 0.1 else '.' if p > 0.05 else '*' if p > 0.01 else '**' if p > 0.001 else '***'
            print('%.3e\t%.3e\t%.3f\t%.4f\t%s' % (summ['betas'][i], summ['se'][i], summ['t'][i], p, stars))



if __name__ == '__main__':
    r = Regressor()
    r.read_csv("bodyfat.csv")
    r.lm("BODYFAT ~ DENSITY + AGE + KNEE")
    r.print_summary()