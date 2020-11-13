# Regression

This project implements multiple linear regression with the ordinary least squares (OLS) loss function ![OLS](https://render.githubusercontent.com/render/math?math=\hat{\beta}=\underset{\beta}{\arg\min}\lVert%20y-X\beta\lVert).  All of the project's code is contained in `regression.py`, and `bodyfat.csv` is provided as a reference dataset.  The output of the functions are meant to mimic those implemented in base R; see the example below.

# Example

```python
import regression

regressor = regression.Regressor()
regressor.read_csv("bodyfat.csv")
regressor.lm("BODYFAT ~ DENSITY + KNEE + AGE + WEIGHT + HEIGHT + ADIPOSITY + NECK + CHEST + ABDOMEN")
regressor.print_summary()
```

```
Residuals:
       Min        1Q    Median        3Q       Max
   -7.6846   -0.3150   -0.0677    0.1879   14.2219

Call:
lm(BODYFAT ~ DENSITY + KNEE + AGE + WEIGHT + HEIGHT + ADIPOSITY + NECK + CHEST + ABDOMEN)

Coefficients:
                Estimate        Std.Error       t value Pr(>|t|)
DENSITY         -3.801e+02      7.152e+00       -53.150 0.0000  ***
KNEE            -4.615e-02      6.090e-02       -0.758  0.4493
AGE             1.031e-02       7.258e-03       1.421   0.1566
WEIGHT          9.911e-03       1.165e-02       0.851   0.3956
HEIGHT          -2.146e-02      2.921e-02       -0.735  0.4632
ADIPOSITY       -7.265e-02      7.632e-02       -0.952  0.3421
NECK            -2.153e-02      5.803e-02       -0.371  0.7109
CHEST           2.916e-02       2.626e-02       1.110   0.2679
ABDOMEN         3.585e-02       2.757e-02       1.300   0.1948
(Intercept)     4.176e+02       9.006e+00       46.375  0.0000  ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 1.345 on 242 degrees of freedom
Multiple R-squared: 0.9784, Adjusted R-squared: 0.9776
F-statistic: 1218.36 on 9 and 242 DF, p-value: 0.0000
```
