#Import required libraries
import keras #library for neural network
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize #machine learning algorithm library

data = pd.read_csv('data/cleaned.csv')

"""for f in data.columns:
    print(data[f].value_counts(dropna=False))
    print(data[f].value_counts(dropna=False).head())
    if data[f].dtypes == 'int64' or data[f].dtypes == 'float64':
        data[f].plot('hist')
        plt.title(f)
        plt.show()
        data.boxplot(column=f, by='crbsi', rot=90)
        plt.show()
    else:
        data[f].value_counts().plot('hist')
        plt.title(f)
        plt.show()"""
for f in data.columns:
    if data[data[f]<0].any().sum() > 0:
        data[f].plot('hist')
        plt.title(f)
        plt.show()
        print(f)

data['age'] = data['age'].abs()
data['total_ventilator_day'] = data['total_ventilator_day'].abs()
data.loc[data.ideal_body_weight<0, 'ideal_body_weight'] = data['ideal_body_weight'].mean()
data.loc[data.net_balance<0, 'net_balance'] = 0

data = data.drop(columns=['Unnamed: 0'])
data.to_csv('data/cleaned.csv')