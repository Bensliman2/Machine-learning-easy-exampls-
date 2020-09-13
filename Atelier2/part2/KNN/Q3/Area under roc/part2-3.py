import pandas
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
url = "/Users/HP/Desktop/S4/Machine Learning/Dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
scoring = 'roc_auc'           
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
print(results.std())
