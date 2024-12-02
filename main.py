import pandas as pd
from data_processing import clean_data
from anomaly_detection import detect_anomalies
from model_training import train_model
from forecasting import forecast
from evaluation import evaluate_model
from custom_algorithms import custom_anomaly_detection

# Основний блок
if __name__ == "__main__":
    # Отримання даних
    data = pd.read_csv('data/input_data.csv')

    # Виявлення аномалій
    anomalies = custom_anomaly_detection(data)
    print(f"Виявлені аномалії: {sum(anomalies)}")

    # Очищення даних
    clean_data = clean_data(data, anomalies)

    # Навчання моделі
    model = train_model(clean_data)

    # Прогнозування
    predictions = forecast(model, clean_data)

    # Оцінка моделі
    evaluate_model(predictions, clean_data)