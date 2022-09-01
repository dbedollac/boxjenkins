from boxjenkins import boxjenkins
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x_array = []
x_array.append(7)
for i in range(100):
    x_array.append(x_array[-1]+np.random.normal())

model = boxjenkins.autoARIMA()
model_fit = model.fit(np.array(x_array))
print(model_fit.params)

df_plot_forecast = pd.DataFrame({'forecast':model_fit.forecast(10)['mean_forecast']})
df_plot = pd.DataFrame({'observed':x_array})
df_plot = pd.concat([df_plot,df_plot_forecast])
df_plot.reset_index(drop=True,inplace=True)

df_plot.plot()
plt.show()
