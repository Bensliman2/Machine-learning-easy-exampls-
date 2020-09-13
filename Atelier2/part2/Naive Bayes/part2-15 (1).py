#Import scikit-learn dataset library
#Import train_test_split function
#Import knearest neighbors Classifier model
#Import scikit-learn metrics module for accuracy calculation
import pandas 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

#Load dataset
url = "/Users/HP/Desktop/S4/Machine Learning/Dataset/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
x = dataframe.values[:, 0:8] 
y = dataframe.values[:,8]
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 70% training and 30% test
#Create nv Classifier
nv = GaussianNB()
#Train the model using the training sets
nv.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = nv.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# hnaya les valeur actual nd predict
df = pandas.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)
