import pandas as pd

def read_data (path) :
    return pd.read_csv(path)

def prepare_multiple_line_data (dataset) :
    dataset = impute_icu_dischage_status(dataset)
    dataset = impute_case_status_in_multiple_line(dataset)
    dataset = normalise_last_data_case_status_in_multiple_line(dataset)
    return dataset

def impute_case_status_in_multiple_line (dataset) :
    # Impute all record which record_day < num_of_record_day to be 0 (Stay in ICU)
    for index,row in dataset.iterrows() :
        if row.record_day < row.num_of_record_day :
            row.case_status = 0
    return dataset

def normalise_last_data_case_status_in_multiple_line (dataset) :
    for index,row in dataset.iterrows() :
        if row.record_day == row.num_of_record_day :
            if row.case_status is 1 :
                if row.icu_discharge_status is 0 :
                    row.case_status = 0
                elif row.icu_discharge_status is 1 or row.icu_discharge_status is 2 :
                    row.case_status = 1
    return dataset

def impute_icu_dischage_status (dataset) :
    # Impute all record with icu_discharge_status from 2 (HAMA) to 1 (Death)
    dataset.icu_discharge_status = dataset.icu_discharge_status.replace(2,1)
    return dataset
