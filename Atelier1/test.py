# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification) 
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix 
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2
# load data 
url = "/Users/HP/Desktop/S4/Machine Learning/Dataset/insurance.csv"
names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
df = pd.read_csv(url, names=names)
df = df.apply(pd.to_numeric, errors='coerce')
#print(df[['sex', 'smoker', 'region']].describe())
array = df.values 
X = array[:,2:3]
Y = array[:,6] 
# feature extraction 
test = SelectKBest(score_func=chi2, k=4) 
fit = test.fit(X, Y) 
# summarize scores 
np.set_printoptions(precision=3) 
print(fit.scores_) 
features = fit.transform(X) 
# summarize selected features 
print(features[0:5,:])

# scatter_matrix(df[['age', 'bmi', 'children', 'charges']]) 
# plt.show()
