from scipy.optimize import minimize
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

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
        var_diff.append(np.nanstd(np.diff(x_array,n=i)))
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

def ComparePolyRoots(x_coeffs,y_coeffs,epsilon=0.2):
    """
    This function compares the real roots of 2 polynomials and returns True if some pair of roots can be considered equal given an epsilon.

    Parameters
    ----------
    x_coeffs: ndarray
        Array with the coefficients of the first polynomial.

    y_coeffs: ndarray
        Array with the coefficients of the second polynomial.

    Returns
    -------
    test: bool
        True if there is a common root, False otherwise.
    """

    x_roots = np.roots(x_coeffs)
    x_real_roots = x_roots[np.isreal(x_roots)]

    y_roots = np.roots(y_coeffs)
    y_real_roots = y_roots[np.isreal(y_roots)]

    test = False
    for x_r in list(x_real_roots):
        if test == True:
            break
        for y_r in list(y_real_roots):
            if abs(x_r-y_r) <= epsilon:
                test = True
                break
    return test

def MaxSigACFlag(x_array):
    """
    This function assess the lags of the ACF and find the last k_lag significantly distinct of 0 with alpha ~5%. The standard deviation is computed according to Bartlett's formula.

    Parameters
    ----------
    x_coeffs: ndarray
        Array with the observed values of the time series.

    Returns
    -------
    q_lag: int
        The last k_lag significantly distinct of 0.

    q_tail: int
        Number of k_lags significantly distinct of 0.

    acf_std: list
        List of the standard deviations of lag_k when q = k-1, beginning with k = 1.
    """
    x_acf = acf(x_array, alpha=0.05)

    x_acf_std = [(l[1] - l[0]) / 4 for l in list(x_acf[1])] # Based on the CI computed by statsmodels that uses 2 standard deviations around each value of the ACF function.

    try:
        q_lag = np.max(np.where(abs(x_acf[0]) > 2*np.array(x_acf_std)))
        q_tail = sum(abs(x_acf[0]) > 2*np.array(x_acf_std))-1
    except:
        q_lag = np.inf
        q_tail = np.inf

    return {'q_lag':q_lag, 'acf_std':x_acf_std[1:], 'q_tail':q_tail}

def MaxSigPACFlag(x_array):
    """
    This function assess the lags of the PACF and find the last k_lag significantly distinct of 0 with alpha ~5%. The standard deviation is computed according to 1/sqrt(len(x)).

    Parameters
    ----------
    x_coeffs: ndarray
        Array with the observed values of the time series.

    Returns
    -------
    p_lag: int
        The last k_lag significantly distinct of 0.

    p_tail: int
        Number of k_lags significantly distinct of 0.

    pacf_std: float
        Standard deviations for every lag in the PACF.
    """
    x_pacf = pacf(x_array, alpha=0.05)
    x_pacf_std = 1/(len(x_array))**0.5
    try:
        p_lag = np.max(np.where(abs(x_pacf[0]) > 2*x_pacf_std))
        p_tail = sum(abs(x_pacf[0]) > 2*x_pacf_std)-1
    except:
        p_lag = np.inf
        p_tail = np.inf

    return {'p_lag':p_lag, 'pacf_std':x_pacf_std, 'p_tail':p_tail}


def GuessModel(x_array):
    """
    This function guess the value of the parameters p and q of an ARIMA model based on the strategy developed by Box and Jenkis, which depends on the interpretation of the ACF and the PACF.

    Parameters
    ----------
    x_coeffs: ndarray
        Array with the observed values of the time series.

    Returns
    -------
    p: int
        The proposed parameter for the p-parameter.
    q: int
        The proposed parameter for the q-parameter.
    """
    max_p = MaxSigPACFlag(x_array)
    max_q = MaxSigACFlag(x_array)

    x_acf = acf(x_array)
    x_acf_diff = abs(np.diff(abs(x_acf),n=1))
    test_acf = x_acf_diff > max_q['acf_std']
    try:
        min_q = np.where(test_acf == True)[0][0] + 1
    except:
        min_q = np.inf

    x_pacf = pacf(x_array)
    x_pacf_diff = abs(np.diff(abs(x_pacf),n=1))
    test_pacf = x_pacf_diff > max_p['pacf_std']

    try:
        min_p = np.where(test_pacf == True)[0][0] + 1
    except:
        min_p = np.inf

    if max_q['q_tail'] > max_p['p_tail']:
        p = max_p['p_lag']
        q = 0
    else:
        if max_q['q_lag'] < min_q+min_p:
            q = max_q['q_lag']
            p = 0
        else:
            p = min_p
            q = min_q

    return {'p':p,'q':q}
