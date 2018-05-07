#Import required libraries
import keras #library for neural network
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize #machine learning algorithm library

#Reading data
data = pd.read_csv('data/cleaned.csv')

data['case_status'] = data['case_status'].astype('object')

print("Describing the data: ",data.describe())
print("Info of the data:",data.info())

print("10 first samples of the dataset:",data.head(10))
print("10 last samples of the dataset:",data.tail(10))

print(data["case_status"].unique())

#Creating train,test and validation data
'''
80% -- train data
20% -- test data
'''
y = data.case_status
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)

#Neural network module
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

y_train=np_utils.to_categorical(y_train,num_classes=3)
y_test=np_utils.to_categorical(y_test,num_classes=3)
print("Shape of y_train",y_train.shape)
print("Shape of y_test",y_test.shape)

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

## The line / model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
print('linear_model')
# Print the accuracy
print("Accuracy: {}".format(model.score(X_test, y_test)))

# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(data=predictions)
prediction_df.to_csv('data/part1_predictions1.csv')

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit the classifier to the training data
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(data=predictions)
prediction_df.to_csv('data/part1_predictions2.csv')

print('LogisticRegression')
# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))
