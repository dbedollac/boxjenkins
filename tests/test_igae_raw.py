from boxjenkins.raw_functions import *
import pandas as pd

df_igae = pd.read_csv('data/igae.csv')
df_igae['igae_t'] = BoxCox(df_igae.igae,GuerreroLambda(df_igae.igae))
anderson_d = AndersonD(df_igae.igae)
