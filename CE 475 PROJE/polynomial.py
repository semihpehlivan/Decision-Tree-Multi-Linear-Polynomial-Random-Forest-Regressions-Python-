import pandas as pd
import matplotlib.pyplot as plt


x_0_100  = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x1','x2','x3','x4','x5','x6'],nrows=100).values

x_0_100_x3  = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x3'],nrows=100).values

x_0_100_x3_nonvalue  = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x3'],nrows=100)


x_100_120= pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x1','x2','x3','x4','x5','x6'],skiprows=range(1,101)).values

x_100_120_x3 = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x3'],skiprows=range(1,101)).values

y_0_100  = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['Y'],nrows=100).values

y_0_100_nonvalue  = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['Y'],nrows=100)

all_dataset = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x1','x2','x3','x4','x5','x6','Y'], nrows=100)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x_0_100_x3,y_0_100)

#2th degree polynom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x_0_100_x3)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_0_100_nonvalue)

# 4th degree polynom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(x_0_100_x3)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y_0_100_nonvalue)
x_poly3_1 = poly_reg3.fit_transform(x_100_120_x3)

y_prediction_100_120= lin_reg3.predict(x_poly3_1)


## Show regression analysis

plt.scatter(x_0_100_x3,y_0_100,color = 'red')
plt.plot(x_0_100_x3,lin_reg2.predict(poly_reg.fit_transform(x_0_100_x3)), color = 'blue')
plt.show()

plt.scatter(x_0_100_x3,y_0_100,color = 'red')
plt.plot(x_0_100_x3,lin_reg3.predict(poly_reg3.fit_transform(x_0_100_x3)), color = 'blue')
plt.show()

