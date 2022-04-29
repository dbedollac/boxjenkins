from boxjenkins.raw_functions import *
import pandas as pd

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
