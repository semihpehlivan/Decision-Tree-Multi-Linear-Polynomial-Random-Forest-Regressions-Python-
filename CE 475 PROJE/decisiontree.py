import pandas as pd


x_0_100  = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x1','x2','x3','x4','x5','x6'],nrows=100).values

x_backward  = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x2','x3','x6'],nrows=100).values

x_100_120 = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x1','x2','x3','x4','x5','x6'],skiprows=range(1,101)).values

x_backward_100_120 = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x2','x3','x6'],skiprows=range(1,101)).values

y_0_100  = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['Y'],nrows=100).values


# Fitting the Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_backward, y_0_100)

# Predicting a new result
y_pred = regressor.predict(x_backward_100_120)
print(y_pred)

#check the model
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = x_backward, y = y_0_100 , cv= 4)
print(accuracies.mean())

