import pickle


def load_data(dataset_type):
    # Load data
    data_path = f"datasets/{dataset_type}/processed_data/"
    x = pickle.load(open(data_path + "x.pkl", "rb"))
    y = pickle.load(open(data_path + "y.pkl", "rb"))
    x_lab_length = pickle.load(open(data_path + "visits_length.pkl", "rb"))

    return x, y, x_lab_length
