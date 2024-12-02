import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def forecast(model, data):
    # Прогнозування на 0.5 інтервалу спостереження
    X_future = np.linspace(data['X'].min(), data['X'].max() + 0.5, 100).reshape(-1, 1)
    X_future_poly = PolynomialFeatures(degree=3).fit_transform(X_future)
    predictions = model.predict(X_future_poly)
    return predictions