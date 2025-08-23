---
---
title: Fine-Tuning GPT-4 to Predict NIFTY50 Prices
description: Learn how to fine-tune GPT-4 with historical NIFTY50 data to predict stock prices and assess its performance through backtesting.
tags:
 - LLM
 - Fine-Tuning
 - GPT-4
 - Azure
 - NIFTY50
 - Stock Market
 - Time Series
---

Language has developed naturally over time, without strict rules in its early stages. This makes it one of the first structured systems shaped by human behavior. By studying language, we can uncover patterns in human thinking and actions. Large Language Models (LLMs) have shown impressive abilities to understand and predict these patterns. In a similar way, business systems like pricing have also evolved organically. Pricing patterns existed long before formal rules were created. If LLMs can predict the next word in a sentence by understanding language patterns, could they also predict the next price using historical data? This article dives into how GPT-4 can be fine-tuned to analyze NIFTY50 price trends and predict future movements, exploring its potential in financial markets.

My focus here is going to be only in Intraday trading. Our system will keep watching the market movement until 2.30 PM. And based on the movement so far, it will make a call or put at 2.30 and square off before the market ends. We'll ask the model to predit a target price.

There will be a stop-loss calculated programatically. Additionally, the cutoff time will be 3.25 PM to close the position. So, if neither target price or stop-loss is reached, the position will be squared off for whatever price at 3.25 PM.

## Data Preperation
The performance of a model is relying only on the data it is trained. So, collecting and preparing the historical Nifty50 data is very crucial. Because if we simply dump the historical price movement to any LLM, it won't make any good other than getting you pay for the GPUs.

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

We need to train the model on daily movements. So, the data should be grouped day-wise. In the dataset the second value is not same for all the indexes. For example, for some dates the first value is available at 9:15:00 but on other days first value is available at 9:15:01. As we don't worry much about the second, we'll unify that to the the minute.

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
<p>5 rows × 375 columns</p>
</div>

<br />

It data is very refreshing. Nifty50 was in its eight thousands in 2015. If you have invested in the index, you would've trippled your money in the past 10 years. This gives us a small problem.




