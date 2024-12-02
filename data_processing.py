import pandas as pd

def clean_data(data, anomalies):
    # Очищення даних від аномалій
    clean_data = data[~anomalies]
    return clean_data