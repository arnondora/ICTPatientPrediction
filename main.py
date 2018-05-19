import data_preprocessing

def main () :
    multiple_line_dataset = data_preprocessing.read_data('./Data/multiple-line-record.csv')
    multiple_line_dataset = data_preprocessing.prepare_multiple_line_data(multiple_line_dataset)

if __name__ == "__main__" :
    main()
