#Import required libraries
import keras #library for neural network
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize #machine learning algorithm library

file = "data/ICU_Project160804 - RRTS Thai Data.xlsx"

#Reading data
datafile = pd.ExcelFile(file)
raw_data = pd.read_excel(datafile, sheet_name=datafile.sheet_names[2])

data = pd.DataFrame(raw_data)

print("Describing the data: ",data.describe())
print("Info of the data:",data.info())
print(data.columns)
print(data.shape)

print("10 first samples of the dataset:",data.head(10))
print("10 last samples of the dataset:",data.tail(10))

data = data[data.columns[data.isnull().mean() < 0.8]]

for f in data.columns:
    if data[f].dtypes == 'object' and ('Yes' == data.ix[0, f] or 'No' == data.ix[0, f]):
        data[f] = data[f].map({'Yes': True, 'No': False})
    elif data[f].dtypes == 'object' :
        data[f] = pd.to_numeric(data[f], errors='coerce')

data.rename(columns={'True_baseline_serum_cr':'baseline'}, inplace=True)
base_index = data['baseline'].index[data['baseline'].apply(np.isnan)]
mdrd_index = data['baseline_serum_creainine_by_mdrd'].index[data['baseline_serum_creainine_by_mdrd'].apply(np.isnan)]
first_index = data['first_available_cr'].index[data['first_available_cr'].apply(np.isnan)]


for base in base_index:
    if base not in mdrd_index and base in first_index:
        data.baseline[base] = data.baseline_serum_creainine_by_mdrd[base]
    elif base in mdrd_index and base not in first_index:
        data.baseline[base] = data.first_available_cr[base]
    elif base not in mdrd_index and base not in first_index:
        if data.first_available_cr[base] > data.baseline_serum_creainine_by_mdrd[base]:
            data.baseline[base] = data.baseline_serum_creainine_by_mdrd[base]
        else:
            data.baseline[base] = data.first_available_cr[base]
    else:
        data.baseline[base] = 0

data = data.drop(columns=['site', 'patient_order','reference_creatinine',
                          'birth_date','baseline_serum_creainine_by_mdrd',
                          'first_available_cr','available_baseline',
                          'date_of_true_baseline_serum_cr','icu',
                          'first_available_cr_date','time_of_icu_admission',
                          'diagnosis', 'date_of_hospital_discharge', 'death_date',
                          'icu_discharge_status', 'date_of_icu_discharge','los_in_icu',
                          'hospital_discharge_status', 'los_in_hospital',
                          'date_of_admission', 'date_of_icu_admission'])

count = 0
for f in data.columns:
    if 'case_status' in f:
        if count == 0:
            miss_index = data[f].index[data[f].apply(np.isnan)]
            for run in miss_index:
                data = data.drop([run])
            old = data[f]
        else:
            new = data[f]
            miss_index = data[f].index[data[f].apply(np.isnan)]
            for run in miss_index:
                new[run] = old[run]
            data[f] = new
            old = data[f]
        count += 1
    elif data[f].dtypes == 'int64' or data[f].dtypes == 'float64':
        data[f].fillna(0, inplace=True)
    elif data[f].dtypes == 'bool' or data[f].dtypes == 'object':
        data[f].fillna(False, inplace=True)


data.to_csv('data/cleaned.csv')
data = pd.read_csv('data/cleaned.csv')

print("Describing the data: ",data.describe())
print("Info of the data:",data.info())
print(data.columns)
print(data.shape)

print("10 first samples of the dataset:",data.head(10))
print("10 last samples of the dataset:",data.tail(10))

print('success')


