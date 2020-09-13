from pandas import read_csv 
from sklearn.preprocessing import Normalizer
import numpy as np
url = "/Users/HP/Desktop/S4/Machine Learning/Dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names=names)
array = df.values 
X = array[:,0:8] 
Y = array[:,8]
transformer = Normalizer().fit(X)
print(transformer) 
Normalizer(copy=True, norm='12') 
print(transformer.transform(X))





 
#result array([[0.8, 0.2, 0.4, 0.4],[0.1, 0.3, 0.9, 0.3],[0.5, 0.7, 0.5, 0.1]])
