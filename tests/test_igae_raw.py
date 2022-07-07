from boxjenkins.raw_functions import *
import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Reading data
df_inpc = pd.read_csv('../data/inp_guerrero.csv')
print(df_inpc.head())

# Selecting lambda with Guerrero method
lambda_g = GuerreroLambda(df_inpc.ipc)
print('lambda_g',lambda_g)

# Transforming data with the selected lambda
df_inpc['ipc_t'] = BoxCox(df_inpc.ipc,lambda_g)
print(df_inpc.head())

# Selecting d with anderson criterio
anderson_d = AndersonD(df_inpc.ipc_t)
print('anderson_d',anderson_d)

# Selecting d wiht Dickey Fuller tests
dickey_d = DickeyD(df_inpc.ipc_t)
print('dickey_d',dickey_d)

# Testing polynomials wiht similar roots
x_coeffs = np.array([1,-3.01,2])
y_coeffs = np.array([1,-3,2])
poly_test = ComparePolyRoots(x_coeffs,y_coeffs,epsilon=0.01)
print('common roots',poly_test)

# Testing to find the maximum lags of the ACF and PAC significantly distinct of 0
df_inpc['ipc_t_diff'] = df_inpc['ipc_t'].diff()
max_q = MaxSigACFlag(df_inpc['ipc_t_diff'].dropna())
max_p = MaxSigPACFlag(df_inpc['ipc_t_diff'].dropna())
print('max_q',max_q['q_lag'])
print('acf_std',max_q['acf_std'])
print('q_tail',max_q['q_tail'])

print('max_p',max_p['p_lag'])
print('pacf_std',max_p['pacf_std'])
print('p_tail',max_p['p_tail'])

acf = acf(df_inpc['ipc_t_diff'].dropna())
pacf = pacf(df_inpc['ipc_t_diff'].dropna())

plt.figure()

plt.subplot(211)
plt.bar(np.arange(len(acf)),acf,label = 'acf')
plt.title('ACF')

plt.subplot(212)
plt.bar(np.arange(len(pacf)),pacf,label = 'pafc')
plt.title('PACF')
plt.show()

# Testing to guess a model using Box the Jenkins strategy.
guess_model_ipc_t_diff = GuessModel(df_inpc['ipc_t_diff'].dropna())
print('model_ipc_t_diff',guess_model_ipc_t_diff)
guess_model_ipc = GuessModel(df_inpc['ipc'].dropna())
print('model_ipc',guess_model_ipc)

# Testing the validation of the assumptions
model_ipc_t_diff_fit = ARIMA(endog=df_inpc['ipc_t'],
                             order=(guess_model_ipc_t_diff['p'],dickey_d,guess_model_ipc_t_diff['q'])
                             ).fit()
print(ValidateAssumptions(model_ipc_t_diff_fit, significance = 0.05))
print('model complies',ValidateAssumptions(model_ipc_t_diff_fit, significance = 0.05)[1])
