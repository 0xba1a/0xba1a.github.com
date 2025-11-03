import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

cwd = Path(__file__).parent.resolve()
data_path = cwd / 'dataset' / 'nifty50_candlestick_data.csv'


class N50PrepAndTest:
    def __init__(self):
        self.dataset_path = data_path
        self.data_set = self.prepare_data()
        self.training_data, self.testing_data = train_test_split(
            self.data_set, test_size=0.2, random_state=42, shuffle=True
        )

    def prepare_data(self):
        # Load your data here
        n50 = pd.read_csv(self.dataset_path)

        n50["datetime"] = pd.to_datetime(n50["Date"] + " " + n50["Time"], format="%d-%m-%Y %H:%M:%S")
        n50.set_index("datetime", inplace=True)
        n50.drop(columns=["Date", "Time", "High", "Low", "Close", "Instrument"], inplace=True)

        market_hours_filter = (n50.index.time >= pd.to_datetime("09:15:00").time()) & (n50.index.time <= pd.to_datetime("15:30:00").time())
        n50 = n50[market_hours_filter]

        n50['date'] = n50.index.date
        n50['time'] = n50.index.strftime('%H:%M')
        n50_pivot = n50.pivot_table(index='date', columns='time', values='Open', aggfunc='first')

        missing_days = n50_pivot.isnull().any(axis=1)
        n50_pivot = n50_pivot[~missing_days]

        n50_pivot["target"] = n50_pivot.loc[:, '14:31':'15:30'].mean(axis=1)

        return n50_pivot

    def get_training_data(self):
        X = self.training_data.loc[:, '09:15':'14:30'].values
        y = self.training_data['target'].values
        return X, y

    def test_profit(self, predict_function: callable):
        X_test = self.testing_data.loc[:, '09:15':'14:30'].values
        y_test = self.testing_data.loc[:, '14:31':'15:25'].values
        exit_price = self.testing_data['15:26'].values

        profit = 0

        for i in range(len(X_test)):
            print(f"\n--------- Testing Day {i+1} ---------")
            action, target_price = predict_function(X_test[i])
            current_price = X_test[i][-1]
            print(f"Action: {action}, Current Price: {current_price}, Target Price: {target_price}")

            if not target_price:
                if action == "buy":
                    target_price = current_price * 1.02
                elif action == "sell":
                    target_price = current_price * 0.98
                elif action == "hold":
                    continue

            exited = False
            for j in range(len(y_test[i])):
                price_at_time = y_test[i][j]

                if action == "buy" and price_at_time >= target_price:
                    profit += (target_price - current_price)
                    exited = True
                    print(f"Exited Buy Position at {price_at_time}")
                    break
                elif action == "sell" and price_at_time <= target_price:
                    profit += (current_price - target_price)
                    exited = True
                    print(f"Exited Sell Position at {price_at_time}")
                    break

            if not exited:
                final_price = exit_price[i]
                # It will take care of negative profit as well
                if action == "buy":
                    profit += (final_price - current_price)
                    print(f"Timedout Buy Position at {final_price}")
                elif action == "sell":
                    profit += (current_price - final_price)
                    print(f"Timedout Sell Position at {final_price}")

        return profit
