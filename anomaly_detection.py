import numpy as np

def detect_anomalies(data):
    # Простий метод виявлення аномалій: Z-score
    threshold = 3
    mean = np.mean(data)
    std_dev = np.std(data)

    z_scores = (data - mean) / std_dev
    anomalies = np.abs(z_scores) > threshold
    return anomalies