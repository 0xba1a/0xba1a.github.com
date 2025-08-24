---
---
title: Fine-Tuning LLMs to Predict NIFTY50 Prices
description: Learn how to fine-tune GPT-4 with historical NIFTY50 data to predict stock prices and assess its performance through backtesting.
date: 2025-08-23
tags:
 - LLM
 - Fine-Tuning
 - Azure
 - NIFTY50
 - Stock Market
 - Time Series
---

# Fine Tuning LLMs to Predict NIFTY50 Price
Language has developed naturally over time, without strict rules in its early stages. This makes it one of the first structured systems shaped by human behavior. By studying language, we can uncover patterns in human thinking and actions. Large Language Models (LLMs) have shown impressive abilities to understand and predict human language. In a similar way, business systems like pricing have also evolved organically. Pricing patterns existed long before formal accounting rules were created. If LLMs can predict the next word in a sentence by understanding language patterns, could they also predict the next price using historical data? This article dives into how LLMs can be trained and fine-tuned to analyze NIFTY50 price trends and predict future movements, exploring its potential in financial markets.

My focus here is only Intraday trading. Because we don't have any price value for NIFTY in the closed hours of market and there can be too many factors affecting the next day opening price. So, trying to predict price across days with current data is nothing but halucination.

Our system will keep watching the market movement until 2.30 PM. And based on the movement so far, it will make a call or put at 2.30 and square off before the market ends. We'll ask the model to predit a target price. There will be a stop-loss calculated programatically. Additionally, the cutoff time will be 3.25 PM to close the position. So, if neither target price or stop-loss is reached, the position will be squared off for whatever price at 3.25 PM.

## Data Preperation
The performance of a model is relying only on the data it is trained. So, collecting and preparing the historical Nifty50 data is very crucial. Simply dumping the historical data to any LLM won't make any good other than getting you pay for the GPUs.

Prepare the environment
```
 $ python3 -m venv .venv
 $ source .venv/bin/activate
 $ pip install pandas
```

I took the 9 years of Nifty-50 candlestick data from this [GitHub repo](https://github.com/sandeepkapri/Nifty50-Minute-Data). It has minute level Open, High, Low and Close information for every market functioning day. We just need one number per minute. So, let's remove everything else other than open.
```
import pandas as pd

candle_stick_data = pd.read_csv("dataset/nifty50_candlestick_data.csv")
candle_stick_data["datetime"] = pd.to_datetime(candle_stick_data["Date"] + " " + candle_stick_data["Time"], format="%d-%m-%Y %H:%M:%S")
candle_stick_data.set_index("datetime", inplace=True)
candle_stick_data.drop(columns=["Date", "Time", "High", "Low", "Close", "Instrument"], inplace=True, errors="ignore")

n50_minute_level_opens = candle_stick_data
n50_minute_level_opens.head()
```
**Output**
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-09 09:15:00</th>
      <td>8285.45</td>
    </tr>
    <tr>
      <th>2015-01-09 09:16:00</th>
      <td>8292.60</td>
    </tr>
    <tr>
      <th>2015-01-09 09:17:00</th>
      <td>8287.40</td>
    </tr>
    <tr>
      <th>2015-01-09 09:18:00</th>
      <td>8294.25</td>
    </tr>
    <tr>
      <th>2015-01-09 09:19:00</th>
      <td>8300.60</td>
    </tr>
  </tbody>
</table>
</div>

<br />

We need to train the model on daily movements. So, the data should be grouped date-wise. In the dataset the second value is not proper. As we don't worry much about the second, we'll unify that to the the minute. Also, remove any data that is beyond typical Indian market hours.

```
market_hours_filter = (n50_minute_level_opens.index.time >= pd.Timestamp('09:15:00').time()) & \
                      (n50_minute_level_opens.index.time <= pd.Timestamp('15:30:00').time())

n50_min_opens = n50_minute_level_opens[market_hours_filter].copy()

n50_min_opens['date'] = n50_min_opens.index.date
n50_min_opens['time'] = n50_min_opens.index.strftime('%H:%M')

n50_daily_opens = n50_min_opens.pivot_table(
    index='date',
    columns='time',
    values='Open',
    aggfunc='first'  # In case there are duplicates, take the first value
)

n50_daily_opens.head()
```
**Output**
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>time</th>
      <th>09:15</th>
      <th>09:16</th>
      <th>09:17</th>
      <th>09:18</th>
      <th>09:19</th>
      <th>09:20</th>
      <th>09:21</th>
      <th>09:22</th>
      <th>09:23</th>
      <th>09:24</th>
      <th>...</th>
      <th>15:20</th>
      <th>15:21</th>
      <th>15:22</th>
      <th>15:23</th>
      <th>15:24</th>
      <th>15:25</th>
      <th>15:26</th>
      <th>15:27</th>
      <th>15:28</th>
      <th>15:29</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-09</th>
      <td>8285.45</td>
      <td>8292.60</td>
      <td>8287.40</td>
      <td>8294.25</td>
      <td>8300.6</td>
      <td>8300.50</td>
      <td>8300.65</td>
      <td>8302.45</td>
      <td>8294.85</td>
      <td>8295.20</td>
      <td>...</td>
      <td>8280.8</td>
      <td>8282.35</td>
      <td>8283.40</td>
      <td>8284.35</td>
      <td>8286.9</td>
      <td>8286.65</td>
      <td>8283.45</td>
      <td>8282.35</td>
      <td>8283.25</td>
      <td>8280.50</td>
    </tr>
    <tr>
      <th>2015-01-12</th>
      <td>8291.35</td>
      <td>8254.20</td>
      <td>8255.25</td>
      <td>8258.15</td>
      <td>8263.2</td>
      <td>8267.45</td>
      <td>8266.05</td>
      <td>8268.80</td>
      <td>8273.85</td>
      <td>8266.75</td>
      <td>...</td>
      <td>8329.5</td>
      <td>8326.55</td>
      <td>8328.05</td>
      <td>8328.05</td>
      <td>8327.2</td>
      <td>8330.20</td>
      <td>8330.90</td>
      <td>8329.95</td>
      <td>8329.95</td>
      <td>8328.85</td>
    </tr>
    <tr>
      <th>2015-01-13</th>
      <td>8346.15</td>
      <td>8355.15</td>
      <td>8348.70</td>
      <td>8344.50</td>
      <td>8342.5</td>
      <td>8340.35</td>
      <td>8339.75</td>
      <td>8340.45</td>
      <td>8333.30</td>
      <td>8326.05</td>
      <td>...</td>
      <td>8304.9</td>
      <td>8305.75</td>
      <td>8306.50</td>
      <td>8307.15</td>
      <td>8308.0</td>
      <td>8308.20</td>
      <td>8308.25</td>
      <td>8307.25</td>
      <td>8305.85</td>
      <td>8308.20</td>
    </tr>
    <tr>
      <th>2015-01-14</th>
      <td>8307.25</td>
      <td>8300.85</td>
      <td>8307.00</td>
      <td>8309.05</td>
      <td>8305.4</td>
      <td>8304.70</td>
      <td>8302.20</td>
      <td>8293.10</td>
      <td>8296.70</td>
      <td>8306.85</td>
      <td>...</td>
      <td>8280.1</td>
      <td>8278.90</td>
      <td>8280.90</td>
      <td>8283.60</td>
      <td>8284.3</td>
      <td>8285.35</td>
      <td>8285.50</td>
      <td>8286.95</td>
      <td>8288.30</td>
      <td>8288.90</td>
    </tr>
    <tr>
      <th>2015-01-15</th>
      <td>8425.20</td>
      <td>8440.45</td>
      <td>8394.35</td>
      <td>8386.05</td>
      <td>8401.1</td>
      <td>8428.00</td>
      <td>8408.25</td>
      <td>8398.00</td>
      <td>8416.70</td>
      <td>8421.95</td>
      <td>...</td>
      <td>8497.6</td>
      <td>8491.80</td>
      <td>8482.05</td>
      <td>8477.25</td>
      <td>8468.0</td>
      <td>8463.80</td>
      <td>8469.05</td>
      <td>8464.80</td>
      <td>8467.25</td>
      <td>8467.45</td>
    </tr>
  </tbody>
</table>
</div>

<br />

This data is very refreshing. Nifty50 was in its eight thousands in 2015. If you have invested in the index, you would've trippled your money in the past 10 years. This gives us a small problem. I try to make the LLM to understand the trend in human price setting behaviour. Whether the price is 8000 or 24000, the trend should be same. But if I pass different prices (tokens), LLM may consider them as different behaviours. This may lead to a situation where LLM will give less importance to the original feature that defines the trend.

So, I decide to pass the price difference in percentage instead of passing the price itself. The idea here is to keep the open price at 9.15 as the reference and calculate the difference in percentage for 9:16. Then using price of 9.16 as reference calculate the different for 9.17. Like this we continue for the whole day with respective to the previous minute price. My assumption is whatever the price is, we humans tend to set the new price relatively.
```
# Calculate percentage price movements within each day
# For each day, calculate percentage change from previous minute
n50_daily_price_movements = n50_daily_opens.pct_change(axis=1, fill_method=None) * 100

# Set the first column (first minute of each day) to 0 as there's no reference price
n50_daily_price_movements.iloc[:, 0] = 0

n50_daily_price_movements.head()
```
**Output**
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>time</th>
      <th>09:15</th>
      <th>09:16</th>
      <th>09:17</th>
      <th>09:18</th>
      <th>09:19</th>
      <th>09:20</th>
      <th>09:21</th>
      <th>09:22</th>
      <th>09:23</th>
      <th>09:24</th>
      <th>...</th>
      <th>15:20</th>
      <th>15:21</th>
      <th>15:22</th>
      <th>15:23</th>
      <th>15:24</th>
      <th>15:25</th>
      <th>15:26</th>
      <th>15:27</th>
      <th>15:28</th>
      <th>15:29</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-09</th>
      <td>0.0</td>
      <td>0.086296</td>
      <td>-0.062707</td>
      <td>0.082656</td>
      <td>0.076559</td>
      <td>-0.001205</td>
      <td>0.001807</td>
      <td>0.021685</td>
      <td>-0.091539</td>
      <td>0.004219</td>
      <td>...</td>
      <td>-0.008453</td>
      <td>0.018718</td>
      <td>0.012678</td>
      <td>0.011469</td>
      <td>0.030781</td>
      <td>-0.003017</td>
      <td>-0.038616</td>
      <td>-0.013279</td>
      <td>0.010866</td>
      <td>-0.033200</td>
    </tr>
    <tr>
      <th>2015-01-12</th>
      <td>0.0</td>
      <td>-0.448057</td>
      <td>0.012721</td>
      <td>0.035129</td>
      <td>0.061152</td>
      <td>0.051433</td>
      <td>-0.016934</td>
      <td>0.033269</td>
      <td>0.061073</td>
      <td>-0.085813</td>
      <td>...</td>
      <td>0.042037</td>
      <td>-0.035416</td>
      <td>0.018015</td>
      <td>0.000000</td>
      <td>-0.010206</td>
      <td>0.036027</td>
      <td>0.008403</td>
      <td>-0.011403</td>
      <td>0.000000</td>
      <td>-0.013205</td>
    </tr>
    <tr>
      <th>2015-01-13</th>
      <td>0.0</td>
      <td>0.107834</td>
      <td>-0.077198</td>
      <td>-0.050307</td>
      <td>-0.023968</td>
      <td>-0.025772</td>
      <td>-0.007194</td>
      <td>0.008394</td>
      <td>-0.085727</td>
      <td>-0.087000</td>
      <td>...</td>
      <td>0.080137</td>
      <td>0.010235</td>
      <td>0.009030</td>
      <td>0.007825</td>
      <td>0.010232</td>
      <td>0.002407</td>
      <td>0.000602</td>
      <td>-0.012036</td>
      <td>-0.016853</td>
      <td>0.028293</td>
    </tr>
    <tr>
      <th>2015-01-14</th>
      <td>0.0</td>
      <td>-0.077041</td>
      <td>0.074089</td>
      <td>0.024678</td>
      <td>-0.043928</td>
      <td>-0.008428</td>
      <td>-0.030103</td>
      <td>-0.109610</td>
      <td>0.043410</td>
      <td>0.122338</td>
      <td>...</td>
      <td>0.078563</td>
      <td>-0.014493</td>
      <td>0.024158</td>
      <td>0.032605</td>
      <td>0.008450</td>
      <td>0.012675</td>
      <td>0.001810</td>
      <td>0.017500</td>
      <td>0.016291</td>
      <td>0.007239</td>
    </tr>
    <tr>
      <th>2015-01-15</th>
      <td>0.0</td>
      <td>0.181005</td>
      <td>-0.546179</td>
      <td>-0.098876</td>
      <td>0.179465</td>
      <td>0.320196</td>
      <td>-0.234338</td>
      <td>-0.121904</td>
      <td>0.222672</td>
      <td>0.062376</td>
      <td>...</td>
      <td>-0.042347</td>
      <td>-0.068255</td>
      <td>-0.114817</td>
      <td>-0.056590</td>
      <td>-0.109116</td>
      <td>-0.049598</td>
      <td>0.062029</td>
      <td>-0.050183</td>
      <td>0.028943</td>
      <td>0.002362</td>
    </tr>
  </tbody>
</table>
</div>

Let's analyze the price movement for insights. Since the true value of an asset (like Nifty50) is often unclear, we use the current and previous prices as proxies—a concept tied to Daniel Kahneman's Anchoring Bias. This bias suggests that sudden price increases are likely to be corrected downward, while sharp decreases are adjusted upward. As a result, the average price movement should ideally converge to zero.
```
# Calculate statistics excluding NaN values and the first column (which is all zeros)
movements_data = n50_daily_price_movements.iloc[:, 1:].values.flatten()  # Exclude first column
movements_data_clean = movements_data[~pd.isna(movements_data)]  # Remove NaN values

print(f"Total data points: {len(movements_data_clean):,}")
print(f"Mean movement: {movements_data_clean.mean():.4f}%")
print(f"Std deviation: {movements_data_clean.std():.4f}%")
print(f"Min movement: {movements_data_clean.min():.4f}%")
print(f"Max movement: {movements_data_clean.max():.4f}%")
```
**Output**
```
Total data points: 849,150
Mean movement: -0.0002%
Std deviation: 0.0409%
Min movement: -2.4480%
Max movement: 6.2991%
```

Our assumption didn't go wrong. The mean value is very close to zero, -0.0002%. If you see the standard deviation, it is around 0.04%. It means in the last 10 years, for more than 68% percentage of the time we quoated the new price with is ±0.04% relative to the current price.

*Note: In our case the ±1σ is 82.69%. It means 82% of times we qouted the new price within ±0.04% of the current price. Similarly Nifty50 movement distribution's ±2σ is 96.01% whereas typical distribution is 95.45%. At 2σ only the price movement converges to normal statistics. The ±3σ is 98.64% but normal is 99.73% - we have more outliers here.*

**This insight reveals that we tend to adopt a more cautious approach under normal circumstances. However, when pushed to the edge, we take significantly higher risks. The difference in the 3σ range is notable. While 4.28% of data points are expected in this range, only 2.63% are present—nearly half are missing. This suggests that when we decide to take risks, we often overextend, taking unnecessary risks about 50% of the time.**

*I'm leaving it as an excercise for you to calculate the ranges yourself.*

Let's clean the data.

1. Fill any NaN with previous value
2. Fix the precision to two decimal degits. *It's safe to have 0.04% as our approximate std.*
```
# Fill NaN values with previous value (forward fill along rows)
n50_daily_price_movements = n50_daily_price_movements.ffill(axis=1)

# Round to 2 decimal places
n50_daily_price_movements = n50_daily_price_movements.round(2)

print(f"Price movements DataFrame shape: {n50_daily_price_movements.shape}")
print(f"NaN values remaining: {n50_daily_price_movements.isna().sum().sum()}")

n50_daily_price_movements.head()
```
**Output**
```
Price movements DataFrame shape: (2273, 375)
NaN values remaining: 0
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>time</th>
      <th>09:15</th>
      <th>09:16</th>
      <th>09:17</th>
      <th>09:18</th>
      <th>09:19</th>
      <th>09:20</th>
      <th>09:21</th>
      <th>09:22</th>
      <th>09:23</th>
      <th>09:24</th>
      <th>...</th>
      <th>15:20</th>
      <th>15:21</th>
      <th>15:22</th>
      <th>15:23</th>
      <th>15:24</th>
      <th>15:25</th>
      <th>15:26</th>
      <th>15:27</th>
      <th>15:28</th>
      <th>15:29</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-09</th>
      <td>0.0</td>
      <td>0.09</td>
      <td>-0.06</td>
      <td>0.08</td>
      <td>0.08</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>-0.09</td>
      <td>0.00</td>
      <td>...</td>
      <td>-0.01</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.03</td>
      <td>-0.00</td>
      <td>-0.04</td>
      <td>-0.01</td>
      <td>0.01</td>
      <td>-0.03</td>
    </tr>
    <tr>
      <th>2015-01-12</th>
      <td>0.0</td>
      <td>-0.45</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>-0.02</td>
      <td>0.03</td>
      <td>0.06</td>
      <td>-0.09</td>
      <td>...</td>
      <td>0.04</td>
      <td>-0.04</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>-0.01</td>
      <td>0.04</td>
      <td>0.01</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>2015-01-13</th>
      <td>0.0</td>
      <td>0.11</td>
      <td>-0.08</td>
      <td>-0.05</td>
      <td>-0.02</td>
      <td>-0.03</td>
      <td>-0.01</td>
      <td>0.01</td>
      <td>-0.09</td>
      <td>-0.09</td>
      <td>...</td>
      <td>0.08</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>-0.01</td>
      <td>-0.02</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>2015-01-14</th>
      <td>0.0</td>
      <td>-0.08</td>
      <td>0.07</td>
      <td>0.02</td>
      <td>-0.04</td>
      <td>-0.01</td>
      <td>-0.03</td>
      <td>-0.11</td>
      <td>0.04</td>
      <td>0.12</td>
      <td>...</td>
      <td>0.08</td>
      <td>-0.01</td>
      <td>0.02</td>
      <td>0.03</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>2015-01-15</th>
      <td>0.0</td>
      <td>0.18</td>
      <td>-0.55</td>
      <td>-0.10</td>
      <td>0.18</td>
      <td>0.32</td>
      <td>-0.23</td>
      <td>-0.12</td>
      <td>0.22</td>
      <td>0.06</td>
      <td>...</td>
      <td>-0.04</td>
      <td>-0.07</td>
      <td>-0.11</td>
      <td>-0.06</td>
      <td>-0.11</td>
      <td>-0.05</td>
      <td>0.06</td>
      <td>-0.05</td>
      <td>0.03</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>


<br />
To further refine the dataset, we will remove high-stress days. These are days where extreme emotions drive price movements, leading to significant variations. Including such days in the dataset might confuse the model, as it could attempt to scale these outliers alongside normal trading days. By excluding these high-stress days, we aim to create a more consistent dataset that better represents typical market behavior.

By removing these high-stress days, we ensure that the dataset focuses on regular market conditions, which are more representative of typical trading patterns. This adjustment will help the model generalize better and avoid overfitting to rare, extreme scenarios.

To identify and filter out high-stress days, we will use the daily standard deviation as a measure of volatility. Days with a standard deviation outside the range of ±2σ (calculated from the mean daily standard deviation) will be considered high-stress days and excluded from the dataset. Here we calculate the std of every day. And put that in a series then calculate mean and std of that series. So, please don't get confused with `std of std`.
```
# Calculate daily standard deviation for each trading day
daily_std = n50_daily_price_movements.std(axis=1)  # std across columns (time) for each day

print(f"Daily std statistics:")
print(f"Mean daily std: {daily_std.mean():.4f}%")
print(f"Std of daily std: {daily_std.std():.4f}%")
print(f"Min daily std: {daily_std.min():.4f}%")
print(f"Max daily std: {daily_std.max():.4f}%")

# Calculate the mean and std of daily standard deviations
mean_daily_std = daily_std.mean()
std_daily_std = daily_std.std()

# Define the acceptable range (±2σ)
lower_bound = mean_daily_std - 2 * std_daily_std
upper_bound = mean_daily_std + 2 * std_daily_std

print(f"\nAcceptable daily std range: {lower_bound:.4f}% to {upper_bound:.4f}%")

# Filter days that fall within ±2σ of mean daily std
days_within_2sigma = (daily_std >= lower_bound) & (daily_std <= upper_bound)

print(f"\nDays analysis:")
print(f"Total days before filtering: {len(n50_daily_price_movements)}")
print(f"Days within ±2σ: {days_within_2sigma.sum()}")
print(f"Days to remove: {len(n50_daily_price_movements) - days_within_2sigma.sum()}")
print(f"Percentage kept: {days_within_2sigma.sum() / len(n50_daily_price_movements) * 100:.2f}%")

# Apply the filter
n50_daily_price_movements_filtered = n50_daily_price_movements[days_within_2sigma]
n50_daily_opens_filtered = n50_daily_opens[days_within_2sigma]

print(f"\nFiltered dataset shape:")
print(f"Price movements: {n50_daily_price_movements_filtered.shape}")
print(f"Daily opens: {n50_daily_opens_filtered.shape}")

# Show some examples of removed days (outliers)
outlier_days = n50_daily_price_movements[~days_within_2sigma]
if len(outlier_days) > 0:
    print(f"\nExamples of removed days (high/low volatility):")
    print(f"Highest volatility day: {daily_std.idxmax()} (std: {daily_std.max():.4f}%)")
    print(f"Lowest volatility day: {daily_std.idxmin()} (std: {daily_std.min():.4f}%)")

n50_daily_price_movements_filtered.head()
```
**Output**
```
Daily std statistics:
Mean daily std: 0.0351%
Std of daily std: 0.0211%
Min daily std: 0.0104%
Max daily std: 0.4735%

Acceptable daily std range: -0.0070% to 0.0772%

Days analysis:
Total days before filtering: 2273
Days within ±2σ: 2221
Days to remove: 52
Percentage kept: 97.71%

Filtered dataset shape:
Price movements: (2221, 375)
Daily opens: (2221, 375)

Examples of removed days (high/low volatility):
Highest volatility day: 2020-03-13 (std: 0.4735%)
Lowest volatility day: 2024-03-02 (std: 0.0104%)
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>time</th>
      <th>09:15</th>
      <th>09:16</th>
      <th>09:17</th>
      <th>09:18</th>
      <th>09:19</th>
      <th>09:20</th>
      <th>09:21</th>
      <th>09:22</th>
      <th>09:23</th>
      <th>09:24</th>
      <th>...</th>
      <th>15:20</th>
      <th>15:21</th>
      <th>15:22</th>
      <th>15:23</th>
      <th>15:24</th>
      <th>15:25</th>
      <th>15:26</th>
      <th>15:27</th>
      <th>15:28</th>
      <th>15:29</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-09</th>
      <td>0.0</td>
      <td>0.09</td>
      <td>-0.06</td>
      <td>0.08</td>
      <td>0.08</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>-0.09</td>
      <td>0.00</td>
      <td>...</td>
      <td>-0.01</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.03</td>
      <td>-0.00</td>
      <td>-0.04</td>
      <td>-0.01</td>
      <td>0.01</td>
      <td>-0.03</td>
    </tr>
    <tr>
      <th>2015-01-12</th>
      <td>0.0</td>
      <td>-0.45</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>-0.02</td>
      <td>0.03</td>
      <td>0.06</td>
      <td>-0.09</td>
      <td>...</td>
      <td>0.04</td>
      <td>-0.04</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>-0.01</td>
      <td>0.04</td>
      <td>0.01</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>2015-01-13</th>
      <td>0.0</td>
      <td>0.11</td>
      <td>-0.08</td>
      <td>-0.05</td>
      <td>-0.02</td>
      <td>-0.03</td>
      <td>-0.01</td>
      <td>0.01</td>
      <td>-0.09</td>
      <td>-0.09</td>
      <td>...</td>
      <td>0.08</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>-0.01</td>
      <td>-0.02</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>2015-01-14</th>
      <td>0.0</td>
      <td>-0.08</td>
      <td>0.07</td>
      <td>0.02</td>
      <td>-0.04</td>
      <td>-0.01</td>
      <td>-0.03</td>
      <td>-0.11</td>
      <td>0.04</td>
      <td>0.12</td>
      <td>...</td>
      <td>0.08</td>
      <td>-0.01</td>
      <td>0.02</td>
      <td>0.03</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>2015-01-15</th>
      <td>0.0</td>
      <td>0.18</td>
      <td>-0.55</td>
      <td>-0.10</td>
      <td>0.18</td>
      <td>0.32</td>
      <td>-0.23</td>
      <td>-0.12</td>
      <td>0.22</td>
      <td>0.06</td>
      <td>...</td>
      <td>-0.04</td>
      <td>-0.07</td>
      <td>-0.11</td>
      <td>-0.06</td>
      <td>-0.11</td>
      <td>-0.05</td>
      <td>0.06</td>
      <td>-0.05</td>
      <td>0.03</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>

This cleansing removed 52 days from the market.

*Most of the blogs and influencers will take only these 52 days. Probably, if we have of all the blogs and youtube transcripts and arrange the dates mentioned in a series, 2σ dates will fall in these 52 days only. You decide whether to make profit in 2221 normal days or you're going to wait for one of those 52 days.*

Finally, let's split the dataset into training and validation sets and store them as separate files. The model will be trained only on the training dataset. The validation set will not be exposed to the model during training or fine-tuning. We'll use the validation dataset to backtest whether the model is makeing any profit for us.

Let's take out every nineth day into validation set. As it is more than seven, the subsequent nineth day will be different day of the week.

```
# Split into training and validation sets
# Every 9th day goes to validation, rest goes to training
total_days = len(n50_daily_price_movements_filtered)

# Create boolean masks for train/validation split
validation_mask = [(i % 9 == 8) for i in range(total_days)]  # Every 9th day (0-indexed, so 8th position)
training_mask = [not val for val in validation_mask]

# Split the datasets
train_price_movements = n50_daily_price_movements_filtered[training_mask]
val_price_movements = n50_daily_price_movements_filtered[validation_mask]

train_daily_opens = n50_daily_opens_filtered[training_mask]
val_daily_opens = n50_daily_opens_filtered[validation_mask]

# Save the datasets to CSV files in the dataset directory
import os

# Create dataset directory if it doesn't exist
os.makedirs('dataset', exist_ok=True)

# Save training datasets
train_price_movements.to_csv('dataset/train_price_movements.csv')
train_daily_opens.to_csv('dataset/train_daily_opens.csv')

# Save validation datasets
val_price_movements.to_csv('dataset/val_price_movements.csv')
val_daily_opens.to_csv('dataset/val_daily_opens.csv')

print("Datasets saved successfully!")
```

## Conclusion
Let's wrap up this blog here. In the upcoming posts, we will explore various models and training methodologies to determine which approach yields the best results. The dataset prepared in this blog will serve as the foundation for all those experiments. You can access the code discussed in this blog in the [n50_dataset_prep](https://github.com/0xba1a/0xba1a.github.com/blob/master/ai_market_prediction/n50_dataset_prep.ipynb) notebook.


