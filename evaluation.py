from sklearn.metrics import mean_squared_error

def evaluate_model(predictions, actual_data):
    mse = mean_squared_error(actual_data['y'], predictions)
    print(f'Mean Squared Error: {mse}')