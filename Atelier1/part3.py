import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("/Users/HP/Desktop/S4/Machine Learning/Dataset/insurance.csv")
print(dataset.describe())
X = dataset[['age','bmi','children']].values
y = dataset['charges'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
#print(coeff_df)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
print(df1)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))