import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def train_model(data):
    # Припустимо, що data має одну ознаку 'X' та цільову змінну 'y'
    X = data[['X']].values
    y = data['y'].values
    # Поліноміальна регресія
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    return model