import pandas as pd
import numpy as np


#Import Data
all_dataset = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x1','x2','x3','x4','x5','x6','Y'], nrows=100)

x_0_100  = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x1','x2','x3','x4','x5','x6'],nrows=100).values

y_0_100  = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['Y'],nrows=100).values

x_100_120 = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x1','x2','x3','x4','x5','x6'],skiprows=range(1,101)).values


#Multiple Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_0_100,y_0_100)

y_pred = regressor.predict(x_100_120)

#BACKWARD ELIMINATION
import statsmodels.formula.api as sm

a = np.append(arr=np.ones((100,1)).astype(int), values = x_0_100, axis=1)
a_l = x_0_100[:,[0,1,2,3,4,5]]
r_ols = sm.OLS(endog = y_0_100, exog = a_l)
r = r_ols.fit()
print(r.summary())


a_l = x_0_100[:,[1,2,3,4,5]]
r_ols = sm.OLS(endog = y_0_100, exog = a_l)
r = r_ols.fit()
print(r.summary())


a_l = x_0_100[:,[1,2,3,5]]
r_ols = sm.OLS(endog = y_0_100, exog = a_l)
r = r_ols.fit()
print(r.summary())


a_l = x_0_100[:,[1,2,5]]
r_ols = sm.OLS(endog = y_0_100, exog = a_l)
r = r_ols.fit()
print(r.summary())

 








