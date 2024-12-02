import numpy as np

def custom_anomaly_detection(data):
    # Власний алгоритм в иявлення аномалій на основі міжквартильного діапазону (IQR)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies = (data < lower_bound) | (data > upper_bound)
    return anomalies