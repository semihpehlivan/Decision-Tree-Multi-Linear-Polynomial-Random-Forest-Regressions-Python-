import pandas as pd

first_100_x = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x2','x3','x6'],nrows=100).values
last_20_x = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['x2','x3','x6'],skiprows=range(1,101)).values
first_100_y  = pd.read_csv('ce475data.csv', encoding='latin1', usecols=['Y'],nrows=100).values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(first_100_x,first_100_y, test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=21,random_state=0)
regressor.fit(x_train, y_train)
y_prediction_100_120 = regressor.predict(last_20_x)
r=regressor.score(x_test,y_test)


print(y_prediction_100_120)
print(r)

from sklearn.metrics import r2_score

r_score=r2_score(y_test,regressor.predict(x_test))
print(r_score)