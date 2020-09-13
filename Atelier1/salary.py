import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix 
import matplotlib.pyplot as plt 
url = "/Users/HP/Desktop/S4/Machine Learning/Dataset/Salary_Data.csv"
names = ['YearsExperience', 'Salary']
df = pd.read_csv(url, names=names)
df = df.apply(pd.to_numeric, errors='coerce')
#print(df.describe())

df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) 
plt.show() 

scatter_matrix(df) 
plt.show()
 
df.hist() 
plt.show() 