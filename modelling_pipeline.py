import pandas as pd
import numpy as np
import json
from itertools import product
from bisect import bisect_left
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from genetic_selection import GeneticSelectionCV
from lightgbm import LGBMClassifier
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from scipy.stats import mode


def preprocessing(data):
    '''align data type and time order'''
    float_list = [
        'bid_price',
        'bid_qty',
        'ask_price',
        'ask_qty',
        'trade_price',
        'sum_trade_1s',
        'bid_advance_time',
        'ask_advance_time',
        'last_trade_time',
    ]

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    for i in float_list:
        data[i] = data[i].astype(float)

    data = data.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
    return data


def check_null(data):
    '''check null values in dataframe'''
    data = data.copy()
    have_null_cols = list(data.columns[data.isnull().any()])
    print('Columns with null values are {}'.format(', '.join(have_null_cols)))
    for i in have_null_cols:
        print('number of rows that column {} is null: {}'.format(i, data[i].isnull().sum()))
        print('null percentage is {}'.format(round(data[i].isnull().sum() / data.shape[0], 2)))

    stat1 = data['sum_trade_1s'][data['last_trade_time'].isnull()].notnull().sum()
    stat2 = data['last_trade_time'][data['sum_trade_1s'].isnull()].notnull().sum()
    stat3 = data['sum_trade_1s'][data['last_trade_time'] >= 1].isnull().sum()
    stat4 = stat3 / data['sum_trade_1s'].isnull().sum()
    print('number of rows sum_trade_1s is not null when last_trade_time is not: {}'.format(stat1))
    print('number of rows last_trade_time is null when sum_trade_1s is not: {}'.format(stat2))
    print('number of rows sum_trade_1s null at last_trade_time > 1: {}, percentage: {}'.format(stat3, round(stat4, 2)))


def fill_null(data):
    '''
    based on the null check and basic logic, most of the sum_trade_1s null value happens when last_trade_time larger
    than 1 sec (in this case sum_trade_1s should be 0). Therefore, we make an assumption that all the sum_trade_1s null
    value could be filled with 0. Based on such assumption, last_trade_time can be filled with last_trade_time of the
    previous record plus a time movement if record interval is smaller than 1 sec.
    '''

    class last_trade_time_filler:
        prev_last_trade_time = None
        prev_timestamp = None

        @classmethod
        def fill(cls, index):
            last_trade_time = data.loc[index, 'last_trade_time']
            timestamp = data.loc[index, 'timestamp']

            if pd.isnull(last_trade_time):
                time_interval = (timestamp - cls.prev_timestamp).microseconds / (1e+6)
                if time_interval <= 1:
                    last_trade_time = cls.prev_last_trade_time + time_interval
                else:
                    last_trade_time = np.nan

            cls.prev_last_trade_time = last_trade_time
            cls.prev_timestamp = timestamp

            return last_trade_time

    data = data.copy()
    data.loc[data['sum_trade_1s'].isnull(), 'sum_trade_1s'] = 0
    data['last_trade_time'] = data.index.map(last_trade_time_filler.fill)
    print('number of null columns is: {} now'.format(len(list(data.columns[data.isnull().any()]))))

    return data


def x_y_split(data):
    label_cols = ['_1s_side', '_3s_side', '_5s_side']
    feature_cols = list(set(data.columns) - set(label_cols))
    y = data[label_cols].copy()
    x = data[feature_cols].copy()

    return x, y


class correlation_filter:
    remove_cols = []

    @classmethod
    def filter(cls, x, threshold=0.99):
        x = x.copy()
        index2col = {i: col for i, col in enumerate(x.columns)}
        corr = np.array(x.corr())
        correlated_pairs = list(zip(*np.where(np.abs(corr) >= threshold)))
        to_be_delete = []
        for i, j in correlated_pairs:
            former = index2col[i]
            latter = index2col[j]
            if former != latter:
                add = True
                for i, del_set in enumerate(to_be_delete):
                    has_intersect = ({former, latter} & del_set) != {}
                    if has_intersect:
                        add = False
                        to_be_delete[i] = del_set | {former, latter}
                if add:
                    to_be_delete.append({former, latter})

        for i in to_be_delete:
            delete_set = i.copy()
            delete_set.pop()
            x = x.drop(list(delete_set), axis=1)
            cls.remove_cols += list(delete_set)

        return x


class feature_eng:
    timestamp = None
    max_lag = 5
    num_window = [5, 10, 20]
    sec_window = [1, 3, 5, 10]
    rolling_sum_cols = []
    rolling_mean_cols = []
    rolling_max_cols = []
    rolling_min_cols = []
    rolling_std_cols = []

    @staticmethod
    def bid_ask_spread(data):
        data['spread'] = data['ask_price'] - data['bid_price']

    @staticmethod
    def bid_ask_qty_comb(data):
        data['bid_ask_qty_total'] = data['ask_qty'] + data['bid_qty']
        data['bid_ask_qty_diff'] = data['ask_qty'] - data['bid_qty']

    @staticmethod
    def trade_price_feature(data):
        data['trade_price_compare'] = 0  # when trade price between current bid and ask price
        data.loc[data['trade_price'] <= data[
            'bid_price'], 'trade_price_compare'] = -1  # when trade price on current bid side
        data.loc[data['trade_price'] >= data[
            'ask_price'], 'trade_price_compare'] = 1  # when trade price on current sell side

        # whether trade price happens on bid side or ask side during the time it happens
        last_trade_timestamp = data['timestamp'] - pd.to_timedelta(data['last_trade_time'], unit='s')
        idx_list = [bisect_left(data['timestamp'], i) for i in list(last_trade_timestamp)]
        trade_price_pos = []
        for i, index in enumerate(idx_list):
            index1 = index
            index2 = index1 + 1 if index1 < data.shape[0] - 1 else index1
            bid1 = data['bid_price'][index1]
            bid2 = data['bid_price'][index2]
            ask1 = data['ask_price'][index1]
            ask2 = data['ask_price'][index2]
            trade_price = data['trade_price'][i]
            if (bid1 <= trade_price <= bid2) or (bid2 <= trade_price <= bid1):
                trade_price_pos.append(-1)  # happen on bid side
            elif (ask1 <= trade_price <= ask2) or (ask2 <= trade_price <= ask1):
                trade_price_pos.append(1)  # happen on sell side
            else:
                trade_price_pos.append(0)  # unknown case
        data['trade_price_pos'] = trade_price_pos

    @staticmethod
    def diff_feature(data):
        for i in set(data.columns) - {'timestamp'}:
            new_name = '{}_diff'.format(i)
            data[new_name] = data[i] - data[i].shift(1)

    @staticmethod
    def up_or_down(data):
        data['up_down'] = 0
        data.loc[data['bid_price_diff'] < 0, 'up_down'] = -1
        data.loc[data['ask_price_diff'] > 0, 'up_down'] = 1

    @staticmethod
    def lag_feature(data, col, lag):
        new_col_name = '{}_lag_{}'.format(col, lag)
        data[new_col_name] = data[col].shift(lag)

    @staticmethod
    def rolling_feature(data, col, window, feature):
        rolling = data[col].rolling(window=window)
        new_col = '{}_rolling_{}_{}'.format(col, feature, window)

        if feature == 'sum':
            data[new_col] = rolling.sum()
        elif feature == 'mean':
            data[new_col] = rolling.mean()
        elif feature == 'max':
            data[new_col] = rolling.max()
        elif feature == 'min':
            data[new_col] = rolling.min()
        elif feature == 'std':
            data[new_col] = rolling.std()
        elif feature == 'mode':
            data[new_col] = rolling.apply(lambda x: mode(x)[0])

    @classmethod
    def basic_features(cls, data):
        data = data.copy()
        cls.timestamp = data['timestamp']

        cls.bid_ask_spread(data)
        cls.bid_ask_qty_comb(data)
        cls.trade_price_feature(data)
        cls.diff_feature(data)
        cls.up_or_down(data)

        data = data.drop('timestamp', axis=1)
        return data

    @classmethod
    def lag_rolling_features(cls, data):
        data = data.copy()

        # get lag and rolling feature based on previous n records
        rolling_cols = set(data.columns) - {'trade_price_compare', 'trade_price_pos'}
        cls.rolling_sum_cols = [i for i in rolling_cols if 'diff' in i or 'up_down' in i]
        cls.rolling_mean_cols = rolling_cols
        cls.rolling_max_cols = [i for i in rolling_cols if 'bid_qty' in i or 'ask_qty' in i]
        cls.rolling_min_cols = [i for i in rolling_cols if 'bid_qty' in i or 'ask_qty' in i]
        cls.rolling_std_cols = rolling_cols

        for col in rolling_cols:
            for lag in range(1, cls.max_lag + 1):
                cls.lag_feature(data, col, lag)

        for col in rolling_cols:
            for num_window in cls.num_window:
                if col in cls.rolling_sum_cols:
                    cls.rolling_feature(data, col, num_window, 'sum')
                if col in cls.rolling_mean_cols:
                    cls.rolling_feature(data, col, num_window, 'mean')
                if col in cls.rolling_max_cols:
                    cls.rolling_feature(data, col, num_window, 'max')
                if col in cls.rolling_min_cols:
                    cls.rolling_feature(data, col, num_window, 'min')
                if col in cls.rolling_std_cols:
                    cls.rolling_feature(data, col, num_window, 'std')

        # get rolling feature based on previous n seconds
        data.index = cls.timestamp
        for col in rolling_cols:
            for sec_window in cls.sec_window:
                sec_window = '{}s'.format(sec_window)
                if col in cls.rolling_sum_cols:
                    cls.rolling_feature(data, col, sec_window, 'sum')
                if col in cls.rolling_mean_cols:
                    cls.rolling_feature(data, col, sec_window, 'mean')
                if col in cls.rolling_max_cols:
                    cls.rolling_feature(data, col, sec_window, 'max')
                if col in cls.rolling_min_cols:
                    cls.rolling_feature(data, col, sec_window, 'min')
                if col in cls.rolling_std_cols:
                    cls.rolling_feature(data, col, sec_window, 'std')
                if col in ['up_down', 'trade_price_compare', 'trade_price_pos']:
                    cls.rolling_feature(data, col, sec_window, 'mode')

        return data

    @staticmethod
    def remove_na(x, y):
        x = x.reset_index(drop=True)
        x = x.dropna()
        y = y.loc[x.index, :].reset_index(drop=True)
        x = x.reset_index(drop=True)
        return x, y


class feature_selection:
    '''feature selection combining feature importance ranking and GA optimization based on random forest model'''

    @classmethod
    def select(cls, x, y):
        rf_imp_features = cls.rf_imp_features(x, y)
        ga_features = cls.GA_features(x, y)
        features = set(rf_imp_features) | set(ga_features)

        return list(features)

    @classmethod
    def rf_imp_features(cls, x, y, top_perc=0.05):
        '''select top features based on feature importance ranking among all the features'''
        feature_imp = cls.rf_importance_selection(x, y)
        perc_threshold = np.percentile(feature_imp['avg_importance'], int((1 - top_perc) * 100))
        features = list(feature_imp.loc[feature_imp['avg_importance'] >= perc_threshold, 'feature'])

        return features

    @staticmethod
    def rf_importance_selection(x, y, iter_time=3):
        feature_imp = pd.DataFrame(np.zeros((x.shape[1], iter_time + 2)))
        feature_imp.columns = ['feature'] + ['importance_{}'.format(i) for i in range(1, iter_time + 1)] + [
            'avg_importance']
        for col in feature_imp.columns:
            feature_imp[col] = list(x.columns)

        for i in range(1, iter_time + 1):
            col = 'importance_{}'.format(i)
            rf = RandomForestClassifier(n_estimators=10, max_depth=8)
            rf.fit(x, y)
            feature_imp_dict = dict(zip(x.columns, rf.feature_importances_))
            feature_imp[col] = feature_imp[col].replace(feature_imp_dict)

        feature_imp['avg_importance'] = feature_imp.iloc[:, 1:-1].mean(axis=1)
        return feature_imp

    @staticmethod
    def GA_features(x, y):
        rf = RandomForestClassifier(max_depth=8, n_estimators=10)
        selector = GeneticSelectionCV(
            rf,
            cv=TimeSeriesSplit(n_splits=4),
            verbose=1,
            scoring="accuracy",
            max_features=80,
            n_population=200,
            crossover_proba=0.5,
            mutation_proba=0.2,
            n_generations=100,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.05,
            tournament_size=3,
            n_gen_no_change=5,
            caching=True,
            n_jobs=-1
        )
        selector = selector.fit(x, y)
        features = x.columns[selector.support_]

        return features


class model:
    lgbm_paramgrid = {
        'learning_rate': np.arange(0.0005, 0.0015, 0.0001),
        'n_estimators': range(800, 2000, 200),
        'max_depth': [3, 4],
        'colsample_bytree': np.arange(0.2, 0.5, 0.1),
        'reg_alpha': [1],
        'reg_lambda': [1]
    }

    @staticmethod
    def random_forest(x, y):
        rf = RandomForestClassifier(n_estimators=200, max_depth=8)
        rf.fit(x, y)
        return rf

    @classmethod
    def lightgbm(cls, x, y):
        keys, vals = list(zip(*cls.lgbm_paramgrid.items()))
        products = list(product(*vals))
        param_comb = [dict(zip(keys, i)) for i in products]

        if len(param_comb) > 1000:
            best_param = cls.GA_tune_lgbm(x, y)
        else:
            best_param = cls.GS_tune_lgbm(x, y)

        lightgbm = LGBMClassifier(**best_param)
        lightgbm.fit(x, y)

        return lightgbm

    @classmethod
    def GA_tune_lgbm(cls, x, y):
        tuner = EvolutionaryAlgorithmSearchCV(
            estimator=LGBMClassifier(),
            params=cls.lgbm_paramgrid,
            scoring="accuracy",
            cv=TimeSeriesSplit(n_splits=4),
            verbose=1,
            population_size=50,
            gene_mutation_prob=0.2,
            gene_crossover_prob=0.5,
            tournament_size=3,
            generations_number=20,
        )
        tuner.fit(x, y)
        return tuner.best_params_

    @classmethod
    def GS_tune_lgbm(cls, x, y):
        tuner = GridSearchCV(
            estimator=LGBMClassifier(),
            param_grid=cls.lgbm_paramgrid,
            scoring="accuracy",
            cv=TimeSeriesSplit(n_splits=4),
            verbose=1,
            n_jobs=-1,
        )
        tuner.fit(x, y)
        return tuner.best_params_


class feature:
    @staticmethod
    def save(features, correlation_remove):
        final = {
            'keep_features': features,
            'correlation_remove': correlation_remove
        }

        with open('features.txt', 'w') as f:
            f.write(json.dumps(final))

    @staticmethod
    def load():
        with open('features.txt', 'r') as f:
            features = f.read()
            features = json.loads(features)

        return features


def train_model(data, target_label):
    data = data.copy()
    data = preprocessing(data)
    check_null(data)
    data = fill_null(data)
    x, y = x_y_split(data)
    x = feature_eng.basic_features(x)
    x = correlation_filter.filter(x)
    x = feature_eng.lag_rolling_features(x)
    x, y = feature_eng.remove_na(x, y)
    y = y[target_label]
    features = feature_selection.select(x, y)
    feature.save(features, correlation_filter.remove_cols)
    lightgbm = model.lightgbm(x[features], y)
    rf = model.random_forest(x[features], y)
    joblib.dump(rf, 'rf.joblib')
    joblib.dump(lightgbm, 'lgbm.joblib')


def predict(data, target_label):
    '''returns both the prediction and the target_label'''
    features = feature.load()['keep_features']
    correlation_remove = feature.load()['correlation_remove']
    data = data.copy()
    data = preprocessing(data)
    data = fill_null(data)
    x, y = x_y_split(data)
    x = feature_eng.basic_features(x)
    x = x.drop(correlation_remove, axis=1)
    x = feature_eng.lag_rolling_features(x)
    x, y = feature_eng.remove_na(x, y)
    y = y[target_label]
    x = x[features]
    lgbm = joblib.load('lgbm.joblib')
    rf = joblib.load('rf.joblib')
    lgbm_predict = lgbm.predict_proba(x)
    rf_predict = rf.predict_proba(x)
    final_predict = (lgbm_predict + rf_predict) / 2
    final_predict = np.argmax(final_predict, axis=1)

    return final_predict, y


if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    target_label = '_5s_side'
    train_model(data, target_label)
    pred, true_val = predict(data, target_label)
