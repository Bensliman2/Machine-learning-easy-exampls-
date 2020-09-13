import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
#Load dataset
url = "/Users/HP/Desktop/S4/Machine Learning/Dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)
# Assign data from first four columns to X variable
X = data.values[:,0:8]
# Assign data from first fifth columns to y variable
y = data.values[:,8]
#data division
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
#training
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)
#prediction
predictions = mlp.predict(X_test)

# qst 3 Confusion Matrix. 
print(confusion_matrix(y_test,predictions))
# qst 3 Classification Report. ferki binatom f terminal wla ma3rftich koli
print(classification_report(y_test,predictions))
