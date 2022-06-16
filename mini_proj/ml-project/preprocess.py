import pandas as pd
from sklearn.preprocessing import StandardScaler


def scaling(data):
    scaler = StandardScaler()

    if type(data) == pd.DataFrame:
        new_data = scaler.fit_transform(data.values)
    else:
        new_data = scaler.fit_transform(data)

    return new_data


def preprocess_text(data):
    for sent in data:
        new_sent = sent.replace(' ', '')

