from sklearn.datasets import load_iris


def load_data():
    # pd.read_csv('./data/csv)
    data, target = load_iris(return_X_y=True, as_frame=True)

    return data, target