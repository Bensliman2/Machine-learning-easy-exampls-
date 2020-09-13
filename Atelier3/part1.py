import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix 
import matplotlib.pyplot as plt 

df = pd.read_csv('/Users/HP/Desktop/S4/Machine Learning/Dataset/CC GENERAL.csv',index_col='CUST_ID') 
df.drop_duplicates(inplace=True)

scatter_matrix(df) 
plt.show()
 
df.hist() 
plt.show() 
