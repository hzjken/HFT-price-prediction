# HFT-price-prediction
A project of using machine learning model (tree-based) to predict instrument price up or down in high frequency trading.

## Project Background
A data science hands-on exercise of a high frequency trading company. 

## Task
To build a model with the given data to predict whether the trading price will go up or down in a short future. (classification problem)

## Data Explanation
### Feature Columns
<b>timestamp</b>  str, datetime string.<br>
<b>bid_price</b>  float, price of current bid in the market.<br>
<b>bid_qty</b>  float, quantity currently available at the bid price.<br>
<b>bid_price</b>  float, price of current ask in the market.<br>
<b>ask_qty</b>  float, quantity currently available at the ask price.<br>
<b>trade_price</b>  float, last traded price.<br>
<b>sum_trade_1s</b>  float, sum of quantity traded over the last second.<br>
<b>bid_advance_time</b>  float, seconds since bid price last advanced.<br>
<b>ask_advance_time</b>  float, seconds since ask price last advanced.<br>
<b>last_trade_time</b>  float, seconds since last trade.<br>
### Labels
<b>_1s_side</b> int<br>
<b>_3s_side</b> int<br>
<b>_5s_side</b> int<br>
Labels indicate what is type of the first event that will happen in the next x seconds, where:<br>
<b>0</b> -- No price change.<br>
<b>1</b> -- Bid price decreased.<br>
<b>2</b> -- Ask price increased.<br>

## Process
### Preprocessing
<b>data type conversion</b>: **`preprocessing()`**<br>
<b>data check</b>: **`check_null()`**<br>
<b>missing value handling</b>: **`fill_null()`**,
based on the null check and basic logic, most of the sum_trade_1s null value happens when last_trade_time larger
than 1 sec (in this case sum_trade_1s should be 0). Therefore, we make an assumption that all the sum_trade_1s null
value could be filled with 0. Based on such assumption, last_trade_time can be filled with last_trade_time of the
previous record plus a time movement if record interval is smaller than 1 sec.
### Feature Engineering
<b>correlation filter</b>: **`correlation_filter.filter()`**, remove columns that are highly correlated to reduce data redundancy.<br>
<b>logical feature engineering</b>: **`feature_eng.basic_features()`**, build up some features based on trading logic.<br>
<b>time-rolling feature engineering</b>: **`feature_eng.lag_rolling_features()`**, build up features by lagging and rolling of time-series.<br>
### Feature Selection
**`feature_selection.select()`**, Hybrid approach of genetic algorithm selection plus feature importance selection.<br>
<b>genetic algorithm selection</b>: **`feature_selection.GA_features()`** <br>
<b>feature importance selection</b>: **`feature_selection.rf_imp_features()`** <br>
### Modelling
Ensemble of lightGBM and random forest model.<br>
<b>random forest</b>: **`model.random_forest()`** <br>
<b>lightGBM</b>: **`model.lightgbm()`** <br>
### Parameter Tuning
Based on search space to decide whether using grid search or genetic search for lightGBM model's parameter tuning.<br>
<b>grid search</b>: **`model.GS_tune_lgbm()`** <br>
<b>genetic search</b>: **`model.GA_tune_lgbm()`** <br>
###Performance
Out-of-sample classfication accuracy is roughly 76-78%, which means its prediction of the short-term future price movement is acceptable.
