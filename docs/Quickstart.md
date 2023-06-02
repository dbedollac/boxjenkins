# Quickstart


```python
import sys
sys.path.append('../')
```

box_jenkins follows a similar framework to the sklearn model API. For this minimal example, you will create an instance of the autoARIMA class and then call its **fit** and **forecast** methods.

The input to box_jenkins.autoARIMA is only an array like object (list, numpy.array, pandas.Series) of numeric data, which must be a regular time series sorted by date.

## Data Reading

We will use the data from the classic example of Airline Passengers.


```python
import pandas as pd
```


```python
df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unique_id</th>
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AirPassengers</td>
      <td>1949-01-01</td>
      <td>112</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AirPassengers</td>
      <td>1949-02-01</td>
      <td>118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AirPassengers</td>
      <td>1949-03-01</td>
      <td>132</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AirPassengers</td>
      <td>1949-04-01</td>
      <td>129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AirPassengers</td>
      <td>1949-05-01</td>
      <td>121</td>
    </tr>
  </tbody>
</table>
</div>



## Fitting the model 


```python
from boxjenkins.boxjenkins import autoARIMA
```


```python
model = autoARIMA()
model_fit = model.fit(df['y'])
```

We can print the parameters of our model with the params attributes:
- boxcox_lambda: Lambda used to apply a Box-Cox transformation to our data in order to stabilize the variance.
- c: Constant level estimated for the series.
- (p,d,q): ARIMA parameters.


```python
print(model_fit.params)
```

    {'boxcox_lambda': 0.1802567849847353, 'c': 0, 'p': 12, 'd': 1, 'q': 0}


## Forecast 

We use the forecast method of the fitted model in order to make a prediction. We must specify the number of periods to be forecasted with the **n_steps** argument. The forecast method will return a dictionary with the mean forecast and the lower and upper values of a confidence interval (there is an alpha parameter to modify the level of significance set by default to .05 that returns a 95% confidence interval).


```python
model_fit.forecast(n_steps =10)
```




    {'mean_forecast': array([444.25185389, 414.65617096, 439.29292227, 477.80818098,
            496.03276244, 556.858998  , 640.13299933, 614.63925348,
            526.91066141, 476.20529218]),
     'upper_forecast': array([478.40425352, 460.66847198, 497.32817242, 548.2305694 ,
            575.31029059, 650.58299576, 751.26737413, 727.36396422,
            629.76104003, 574.07829861]),
     'lower_forecast': array([412.12429262, 372.48100674, 386.92999042, 414.97931524,
            425.94180172, 474.50267772, 542.85008241, 516.65343005,
            438.25440623, 392.45200921])}



## Plots 


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

Plotting the mean forecast:


```python
df_plot_forecast = pd.DataFrame({'forecast':model_fit.forecast(10)['mean_forecast']})
df_plot = pd.DataFrame({'observed':df['y'].dropna()})
df_plot = pd.concat([df_plot,df_plot_forecast])
df_plot.reset_index(drop=True,inplace=True)

df_plot.plot()
plt.show()
```


    
![png](output_19_0.png)
    


Plotting the forecast with a confidence interval:


```python
df_plot_forecast = pd.DataFrame(model_fit.forecast(10, alpha = 0.05))
df_plot = pd.DataFrame({'observed':df['y'].dropna()})
df_plot = pd.concat([df_plot,df_plot_forecast]).reset_index(drop = True)
df_plot.reset_index(inplace = True)
df_plot = df_plot.melt(id_vars = 'index')
df_plot['type'] = df_plot.variable.apply(lambda x: x if(x == 'observed') else 'forecast')
sns.lineplot(data = df_plot,
             x = 'index',
             y = 'value',
             hue = 'type'
            )
plt.show()
```


    
![png](output_21_0.png)
    



```python

```
