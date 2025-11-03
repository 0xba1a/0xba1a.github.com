from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))
from n50_prep_and_test import N50PrepAndTest

MODEL: LinearRegression = None


def standardize(X, y, predict=True):
    matrix = np.column_stack((X, y))

    if not predict:
        # Remove any day where max - min is more than 5% of median price to avoid high swing days
        matrix = matrix[(matrix.max(axis=1) - matrix.min(axis=1)) <= 0.05 * np.median(matrix[:, :-1])]

    scaler = StandardScaler()
    matrixT = matrix.transpose()
    matrixT_scaled = scaler.fit_transform(matrixT)
    matrix_scaled = matrixT_scaled.transpose()
    return matrix_scaled[:, :-1], matrix_scaled[:, -1]


def predict_action(input_data):
    global MODEL

    # Convert to two dimensional array before passing to standardize
    input_data_std, _ = standardize([input_data], [np.array([0])], predict=True)
    predicted_price_std = MODEL.predict(input_data_std)[0]

    last_input_price = input_data[-1]
    predicted_price = predicted_price_std * np.std(input_data) + np.mean(input_data)

    # Keeping a 1% threshold for action
    profit_threshold = 0.00
    if predicted_price > last_input_price * (1 + profit_threshold):
        return "buy", predicted_price
    elif predicted_price < last_input_price * (1 - profit_threshold):
        return "sell", predicted_price
    else:
        return "hold", None


def main():
    global MODEL

    prep_and_test = N50PrepAndTest()
    X_train, y_train = prep_and_test.get_training_data()

    X_train_std, y_train_std = standardize(X_train, y_train)

    model = LinearRegression()
    model.fit(X_train_std, y_train_std)

    MODEL = model

    profit = prep_and_test.test_profit(predict_action)
    print(f"Total Profit from Linear Regression Model: {profit}")


if __name__ == "__main__":
    main()
