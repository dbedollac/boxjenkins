from scipy.optimize import minimize
import numpy as np
from statsmodels.tsa.stattools import adfuller

def BoxCox(x,l:int):
    """
    This function returns a Box-Cox power transformation of the series x. The transformation is defined as follows: x^l if l!=0 and log(x) if l=0.

    Parameters
    ----------
    x: ndarray
        Input array. Must be positive 1-dimensional. Must not be constant.

    l: int
        Lambda to be applied in the formula of the transformation.

    Returns
    -------
    x_t: ndarray
        Transformed series.
    """
    if np.nanmin(x) <=0:
        raise Exception('Data must be positive.')
    if l == 0:
        return np.log(x)
    else:
        return x**l

def GuerreroLambda(x_array,n_groups:int=2,bounds_list=[-1,1]):
    """
    This function applies Guerrero's (1993) method to select and return the lambda which minimises the coefficient of variation for subseries of x.

    Parameters
    ----------
    x: ndarray
        Input array. Must be positive 1-dimensional. Must not be constant.

    n_groups: int
        Seasonal periodicity of the data in integer form. Must be an integer >= 2. Default is 2.

    bounds: list
        Lower and upper bounds used to restrict the feasible range when solving for the value of lambda. Default [-1,1].

    Returns
    -------
    lambda: int
        Optimal lambda.
    """
    def f_metricsCV_per_group(x,l):
        groups = np.array_split(x,n_groups)
        metrics = []
        for g in groups:
            m_g = np.nanmean(g)
            s_g = np.nanstd(g)
            metrics.append(s_g/m_g**(1-l))
        CV = np.nanstd(metrics)/np.nanmean(metrics)
        return CV

    g_lambda = minimize(fun = lambda l: f_metricsCV_per_group(x_array,l),
                        bounds = [(bounds_list[0],bounds_list[1])],
                        x0=0
                        )
    return g_lambda['x'][0]

def AndersonD(x_array):
    """
    This function applies  Anderson (1976, p. 116) criteria to find the required value for the parameter d in an ARIMA model.

    Parameters
    ----------
    x: ndarray
        Input array. Must be positive 1-dimensional. Must not be constant.

    Returns
    -------
    d: int
        Recommended value for the parameter d.
    """
    var_diff = []  # list fo variances with differentes lags
    for i in range(4):
        var_diff.append(np.nanvar(np.diff(x_array,n=i)))

    return var_diff.index(min(var_diff))

def DickeyD(x_array):
    """
    This function applies [Dickey et al., 1986], criteria to find the required value for the parameter d in an ARIMA model. States the initial value of d=0 and it applies ADF tests until the null hypothesis is rejected.

    Parameters
    ----------
    x: ndarray
        Input array. Must be positive 1-dimensional. Must not be constant.

    Returns
    -------
    d: int
        Recommended value for the parameter d.
    """
    d = 0
    pvalue = adfuller(np.diff(x_array,n=d))[1]
    while pvalue >= 0.05:
        d+=1
        adf_t = adfuller(np.diff(x_array,n=d))
        pvalue = adf_t[1]
    return d
