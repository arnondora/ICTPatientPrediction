import data_preprocessing
import model_trainer

def main () :
    multiple_line_dataset = data_preprocessing.read_data('./Data/multiple-line-record.csv')
    multiple_line_dataset = data_preprocessing.impute_boolean(multiple_line_dataset)
    multiple_line_dataset = data_preprocessing.prepare_multiple_line_data(multiple_line_dataset)
    multiple_line_dataset = data_preprocessing.drop_columns(multiple_line_dataset, ['birth_date', 'date_of_admission', 'date_of_hospital_discharge', 'date_of_icu_admission', 'date_of_icu_discharge', 'date_of_true_baseline_serum_cr', 'death_date', 'first_available_cr_date', 'move_out_time', 'stop_rrt_date', 'time_of_icu_admission'])
    multiple_line_dataset = multiple_line_dataset.fillna(method='backfill')

    print (multiple_line_dataset.isnull().sum())
    # Prepare Feature Space and Label
    multiple_line_features = data_preprocessing.impute_data(multiple_line_dataset.drop(columns=['case_status', 'available_baseline', 'diagnosis', 'type_of_vascular_access', 'internal_jugular_lr', 'dialyzer', 'complication_ihd', 'type_of_vascular_access_sled', 'dialyzer_sled', 'internal_jugular_lr_crrt', 'type_solution', 'site', 'indication', 'vasopressor_type', 'site_of_vascular_access', 'site_of_vascular_access_sled', 'patient_number', 'patient_order', 'hospital_discharge_status', 'icu_discharge_status']))
    multiple_line_label = multiple_line_dataset.case_status

    # multiple_line_dataset.to_csv('./Data/multiple-line-record-transform.csv', sep=",", na_rep='', index=False)

    model = model_trainer.build_gb_model(multiple_line_features, multiple_line_label)
if __name__ == "__main__" :
    main()


# gender, ht, dm, arterial_ph, chronic_health_points, rs, renal, aki_max_staging_from, cr_diagnosis_from_day, cause_of_aki, reimbursement, baseline_serum_creainine_by_mdrd, age, record_day, creatinine, urine_output_cal
