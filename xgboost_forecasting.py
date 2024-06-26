# -*- coding: utf-8 -*-
"""forecasting_xboost.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uGKKtHiOF7doH0-HjlvY9b4lpJzhyPn8
"""



!pip install lightgbm==3.3.5 xgboost

!git clone https://github.com/panambY/Hourly_Energy_Consumption.git

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')

pjme = pd.read_csv('./Hourly_Energy_Consumption/data/PJME_hourly.csv', index_col=[0], parse_dates=[0])

# color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
# _ = pjme.plot(style='.', figsize=(15,5), color=color_pal[0], title='PJM East')

split_date = '01-Jan-2015'
pjme_train = pjme.loc[pjme.index <= split_date].copy()
pjme_test = pjme.loc[pjme.index > split_date].copy()

# _ = pjme_test \
#     .rename(columns={'PJME_MW': 'TEST SET'}) \
#     .join(pjme_train.rename(columns={'PJME_MW': 'TRAINING SET'}), how='outer') \
#     .plot(figsize=(15,5), title='PJM East', style='.')

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week

    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

X_train, y_train = create_features(pjme_train, label='PJME_MW')
X_test, y_test = create_features(pjme_test, label='PJME_MW')

reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False) # Change verbose to True if you want to see it train

# _ = plot_importance(reg, height=0.9)

pjme_test['MW_Prediction'] = reg.predict(X_test)
pjme_all = pd.concat([pjme_test, pjme_train], sort=False)

# _ = pjme_all[['PJME_MW','MW_Prediction']].plot(figsize=(15, 5))

# # Plot the forecast with the actuals
# f, ax = plt.subplots(1)
# f.set_figheight(5)
# f.set_figwidth(15)
# _ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
#                                               style=['-','.'])
# ax.set_xbound(lower='01-01-2015', upper='02-01-2015')
# ax.set_ylim(0, 60000)
# plot = plt.suptitle('January 2015 Forecast vs Actuals')

# f, ax = plt.subplots(1)
# f.set_figheight(5)
# f.set_figwidth(15)
# _ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
#                                               style=['-','.'])
# ax.set_ylim(0, 60000)
# ax.set_xbound(lower='07-01-2015', upper='07-08-2015')
# plot = plt.suptitle('First Week of July Forecast vs Actuals')

mean_squared_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test['MW_Prediction'])

mean_absolute_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test['MW_Prediction'])

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test['MW_Prediction'])

