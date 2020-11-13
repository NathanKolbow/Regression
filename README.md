# Regression

This project implements multiple regression with the ordinary least squares (OLS) loss function ![OLS](https://render.githubusercontent.com/render/math?math=\hat{\beta}=\underset{\beta}{\arg\min}\lVert%20y-X\beta\lVert).  All of the project's code is contained in `regression.py`, and `bodyfat.csv` is provided as a reference dataset.  The output of the functions are meant to mimic those implemented in base R; see the example below.

# Example

```python
import regression

regressor = regression.Regressor()
regressor.read_csv("bodyfat.csv")
regressor.lm("BODYFAT ~ DENSITY + KNEE")
regressor.print_summary()
```

```
Residuals:
       Min        1Q    Median        3Q       Max
   -8.1095   -0.2254   -0.1058    0.0862   15.3660

Call:
lm(BODYFAT ~ DENSITY + KNEE)

Coefficients:
                Estimate        Std.Error       t value Pr(>|t|)
DENSITY         -3.974e+02      4.527e+00       -87.795 0.0000  ***
KNEE            7.936e-02       3.572e-02       2.222   0.0272  *
(Intercept)     4.354e+02       5.591e+00       77.873  0.0000  ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 1.406 on 249 degrees of freedom
Multiple R-squared: 0.9768, Adjusted R-squared: 0.9766
F-statistic: 5236.28 on 2 and 249 DF, p-value: 0.0000
```
