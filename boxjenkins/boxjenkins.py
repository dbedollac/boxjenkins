"""Main module."""
from scipy.optimize import minimize
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.diagnostic import acorr_ljungbox
import itertools
import warnings


class autoARIMA:
    """
    This function selects automatically an ARIMA model for a given univariate time series. The selected model will fulfill the next assumptions:
    - Mean of residuals statistically equal to zero.
    - Residuals described by white noise process.
    - Roots of the lag polynomials out of the unit circle.
    - Lag polynomials without approximately common factors.
    - Principle of parsimony.

    This function assumes that T(x_array) ~ ARIMA(p,d,q) with a constant trend c, where T() is a Box Cox transformation.

    Parameters
    ----------
    x_array: ndarray
        Array with the observed values of the time series.
    maxp: int
        Maximum value of lags for the AR part of the model. If None, the maximum value will be the last k_lag of the PACF significantly distinct of 0. Default is None.
    maxq: int
        Maximum value of lags for the MA part of the model. If None, the maximum value will be the last k_lag of the ACF significantly distinct of 0. Default is None.
    maxd: int
        Maximum number of times that the difference operator can be applied. Default is 3.
    boxcox_transformation: bool
        Whether or not the time series will be transformed in order to stabilize the variance. Default is True.
    anderson_diff: bool
        Whether or not the criteria of Anderson will be used to propose an initial value for d. Default is False.
    guessmodel: bool
        Whether or not the ACF and the PACF functions will be used to propose initial values for p and q based on Box and Jenkins strategy. Default is True.

    Returns
    -------
    boxcox_lambda: float
        The selected lambda for the Box-Cox transformation.
    c: float
        Intercept for model the level fo the time series.
    p: int
        The selected value for the p-parameter.
    d: int
        The selected value for the d-parameter.
    q: int
        The selected value for the q-parameter.
    model: object
        An ARIMAResults class from the statsmodels package. The object is obtained by fitting the T(x_array) with the selected values p,d,q using statsmodels.
    eval: dict
        A dictionary with the key statistics to demonstrate the fulfillment of the mentioned assumptions:
         - mean_test includes the stat value of a t-test for mean = 0 and its p-value.
         - wn_test includes the stat value of a Ljung-Box test, the lag used and its p-value.
         - polynomial_roots includes the roots of the AR and MA polynomials.
    complies: bool
        Whether or not the selected model complies with all the assumptions.
    """
    def __init__(self,maxp = None, maxq = None,maxd = 3, boxcox_transformation = True, anderson_diff = False, guessmodel = True):
        self.maxp = maxp
        self.maxq = maxq
        self.maxd = maxd
        self.boxcox_transformation = boxcox_transformation
        self.anderson_diff = anderson_diff
        self.guessmodel = guessmodel

    def fit(self,x_array):
        maxp = self.maxp
        maxq = self.maxq
        maxd = self.maxd
        boxcox_transformation = self.boxcox_transformation
        anderson_diff = self.anderson_diff
        guessmodel = self.guessmodel

        # Variance stabilization with a Box-cox transfromation
        if boxcox_transformation:
            boxcox_lambda = GuerreroLambda(x_array)
        else:
            boxcox_lambda = 1
        x_array_T = BoxCox(x_array, boxcox_lambda)

        # Level stabilization applying the difference operator
        if anderson_diff:
            d = AndersonD(x_array_T)
        else:
            d = 0
        d = min(d, maxd)
        x_array_T_diff = np.diff(x_array_T, n=d)
        dickey_d = DickeyD(x_array_T_diff, maxd - d)
        d = d + dickey_d

        # Selecting inital values for p,q
        if guessmodel:
            guess = GuessModel(x_array_T_diff)
            p_init = guess['p']
            q_init = guess['q']
        else:
            p_init = 0
            q_init = 0

        # Validating assumptions and iteration of p,q values if needed
        test_assumptions_result = False
        if maxp == None:
            maxp = MaxSigPACFlag(x_array_T_diff)['p_lag']

        if maxq == None:
            maxq = MaxSigACFlag(x_array_T_diff)['q_lag']

        tested_models = []
        tested_models_aic = []
        while (test_assumptions_result == False and len(tested_models) < maxp * maxq):
            minp_i = np.nanmax([p_init - 1, 0])
            minq_i = np.nanmax([q_init - 1, 0])
            maxp_i = np.nanmin([p_init + 1, maxp])
            maxq_i = np.nanmin([q_init + 1, maxq])

            models_to_test = [(minp_i, minq_i), (min(minp_i + 1, maxp_i), minq_i), (minp_i, min(minq_i + 1, maxq_i)),
                              (min(minp_i + 2, maxp_i), minq_i), (min(minp_i + 1, maxp_i), min(minq_i + 1, maxq_i)),
                              (min(minp_i, maxp_i), min(minq_i + 2, maxq_i)),
                              (min(minp_i + 2, maxp_i), min(minq_i + 1, maxq_i)),
                              (min(minp_i + 1, maxp_i), min(minq_i + 2, maxq_i)),
                              (min(minp_i + 2, maxp_i), min(minq_i + 2, maxq_i))
                              ]
            models_to_test = list(set(models_to_test) - set(tested_models))
            models_to_test_aic = []
            if len(models_to_test) > 0:

                for p, q in models_to_test:
                    try:
                        warnings.filterwarnings("ignore")
                        model = ARIMA(endog=x_array_T,
                                      order=(p, d, q),
                                      enforce_stationarity=True,
                                      enforce_invertibility=True
                                      )
                        model_fit = model.fit()
                        test_assumptions = ValidateAssumptions(model_fit, significance=0.05)
                        test_assumptions_result = test_assumptions[1]
                        tested_models_aic.append(model_fit.aic)
                        models_to_test_aic.append(model_fit.aic)
                    except:
                        tested_models_aic.append(np.nan)
                        models_to_test_aic.append(np.nan)

                    tested_models.append((p, q))
                    if test_assumptions_result == True:
                        break
                if test_assumptions_result == False:
                    model_pivot = models_to_test[models_to_test_aic.index(np.nanmin(models_to_test_aic))]
                    p_init = model_pivot[0]
                    q_init = model_pivot[1]
            else:
                for m in itertools.product(range(maxp + 1), range(maxq + 1)):
                    m_test = m in tested_models
                    if m_test == False:
                        p_init = m[0]
                        q_init = m[1]
                        break
            # print(tested_models)
        if test_assumptions_result == False:
            model_params = tested_models[tested_models_aic.index(np.nanmin(tested_models_aic))]
            p = model_params[0]
            q = model_params[1]
            model = ARIMA(endog=x_array_T,
                          order=(p, d, q),
                          enforce_stationarity=True,
                          enforce_invertibility=True
                          )
            model_fit = model.fit()
            test_assumptions = ValidateAssumptions(model_fit, significance=0.05)

        try:
            c = model_fit.params['const']
        except:
            c = 0

        class model_fit_outcome:
            def __init__(self):
                self.params = {'boxcox_lambda': boxcox_lambda,
                                'c': c,
                                'p': p,
                                'd': d,
                                'q': q
                               }
                self.boxcox_lambda = boxcox_lambda
                self.c = c
                self.p = p
                self.d = d
                self.q = q
                self.model_fit = model_fit
                self.eval = test_assumptions[0]
                self.complies = test_assumptions[1]
                self.fitted_values = x_array

            def forecast(self, n_steps, alpha=0.05):
                """
                This function receives an object from the autoARIMA function and make a prediction given a number of steps forward.
                It obtains a mean prediction and bounds of a confidence interval given an alpha. The predictions are computed by
                multiplying a factor to the original forecast in order tu get an unbiased prediction, as proposed by Guerrero (2009).

                Parameters
                ----------
                model: object
                    An object from the autoARIMA function.
                n_steps: int
                    Number of steps to forecast forward.
                alpha: float
                    Signficance level for the confidence interval.

                Returns
                -------
                mean_forecast: ndarray
                    The forecast for the mean value.
                upper_forecast: ndarray
                   The forecast for the upper bound of the confidence interval.
                lower_forecast: ndarray
                   The forecast for the lower bound of the confidence interval.
                """
                l = self.boxcox_lambda
                forecast_init = model_fit.get_forecast(n_steps)
                sigma = forecast_init.se_mean
                t_forecast = forecast_init.predicted_mean
                ci = forecast_init.conf_int(alpha)

                if l < 1:
                    unbiased_factor = 2 * (l - 1) / l
                    unbiased_factor = unbiased_factor * t_forecast ** -2
                    unbiased_factor = unbiased_factor * sigma ** 2
                    unbiased_factor = (0.5 + (1 - unbiased_factor) ** 0.5 / 2) ** (1 / l)
                else:
                    unbiased_factor = 1
                if l >= 0:
                    return {'mean_forecast': unbiased_factor * BoxCoxInv(t_forecast, l),
                            'upper_forecast': unbiased_factor * BoxCoxInv(ci[ci.columns[1]], l),
                            'lower_forecast': unbiased_factor * BoxCoxInv(ci[ci.columns[0]], l)
                            }
                else:
                    return {'mean_forecast': unbiased_factor * BoxCoxInv(t_forecast, l),
                            'upper_forecast': unbiased_factor * BoxCoxInv(ci[ci.columns[0]], l),
                            'lower_forecast': unbiased_factor * BoxCoxInv(ci[ci.columns[1]], l)
                            }

        return model_fit_outcome()


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

def BoxCoxInv(x,l:int):
    """
    This function returns the inverse of a Box-Cox power transformation of the series x. The inverse transformation is defined as follows: x^(1/l) if l!=0 and exp(x) if l=0.

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
        return np.exp(x)
    else:
        return x**(1/l)

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

def DickeyD(x_array,maxd=3):
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
    while (pvalue >= 0.05 and d<maxd):
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
    x_array: ndarray
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
    x_array: ndarray
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
    This function guess the value of the parameters p and q of an ARMA model based on the strategy developed by Box and Jenkis, which depends on the interpretation of the ACF and the PACF.

    Parameters
    ----------
    x_array: ndarray
        Array with the observed values of the time series.

    Returns
    -------
    p: int
        The proposed value for the p-parameter.
    q: int
        The proposed value for the q-parameter.
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


def ValidateAssumptions(model_fit, significance = 0.05):
    """
    Given a fit of an ARIMA model, this function validate the next assumptions:
    - Mean of residuals statistically equal to zero.
    - Residuals described by white noise process.
    - Roots of the lag polynomials out of the unit circle.
    - Lag polynomials without approximately common factors.

    Parameters
    ----------
    model_fit: object
        ARIMAResults object from a fit of an ARIMA model using statsmodels.

    Returns
    -------
    eval: dict
        A dictionary with the key statistics to demonstrate the fulfillment of the mentioned assumptions:
         - mean_test includes the stat value of a t-test for mean = 0 and its p-value.
         - wn_test includes the stat value of a Ljung-Box test, the lag used and its p-value.
         - polynomial_roots includes the roots of the AR and MA polynomials.
    validation: bool
        True if all assumptions are fulfilled, False otherwise.
    validation_list: list
        A list of bool values to describe whether or not the model fulfill each one of the assumptions.
    """
    validation_list = []
    polynomial_ar = model_fit.polynomial_ar
    polynomial_ma = model_fit.polynomial_ma
    residuals = model_fit.resid

    #Testing residuals mean equal to 0
    mean_test = DescrStatsW(residuals).ttest_mean()
    mean_test = {'tstat':mean_test[0],
                 'pvalue':mean_test[1]
                 }
    validation_list.append(mean_test['pvalue'] >= significance)

    #Testing Residuals described by white noise process
    wn_test = acorr_ljungbox(residuals,
                             lags=min(24,len(residuals)//5),
                             model_df = len(polynomial_ar)-1 + len(polynomial_ma)-1
                             )
    wn_test = {'lb_stat':wn_test.lb_stat.values[-1],
               'pvalue': wn_test.lb_pvalue.values[-1],
               'lag': len(wn_test.lb_pvalue.values)
               }
    validation_list.append(wn_test['pvalue'] >= significance)

    #The roots of the lag polynomials are out of the unit circle by default when an ARIMA model is fitted with statsmodels.
    validation_list.append(True)

    #Testing lag polynomials without approximately common factors.
    polynomial_roots = {'arroots': model_fit.arroots,
                        'maroots': model_fit.maroots
                        }
    pol_test = ComparePolyRoots(polynomial_ar,polynomial_ma)
    validation_list.append(not pol_test)

    #Outputs
    validation = (np.prod(np.array(validation_list)) == 1)
    eval ={'mean_test':mean_test,'wn_test':wn_test,'polynomial_roots':polynomial_roots}

    return  eval, validation, validation_list
