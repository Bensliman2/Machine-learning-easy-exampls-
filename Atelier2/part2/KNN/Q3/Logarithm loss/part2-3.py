import pandas
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
url = "/Users/HP/Desktop/S4/Machine Learning/Dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model =  MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = classification_report(Y_test, predicted)
print(matrix)
