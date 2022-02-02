#!/usr/bin/env python
# coding: utf-8

### Package Install


# Programming
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Modelling
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from fbprophet import Prophet
from sklearn.model_selection import ParameterGrid
import xgboost as xgb


# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


### Data Preprocessing

# Import Data
feature = pd.read_csv('features.csv', index_col = 'Date', parse_dates = True)            .drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'], axis = 1)
store = pd.read_csv('stores.csv')
store.Type = [0 if i == 'A' else 1 for i in store.Type]
trainall = pd.read_csv('train_all.csv').drop(['Unnamed: 0','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'], axis = 1)


# merge training and feature dataset
trainall = trainall.groupby(['Store','Date']).agg({'Weekly_Sales':'sum'}).reset_index()
trainall.Date = pd.to_datetime(trainall.Date)
training = trainall.merge(feature, 'inner', on = ['Store', 'Date']).merge(store, 'inner', on = 'Store').rename(columns = {'IsHoliday_x':'IsHoliday'}).set_index('Date')


#renaming datetime/target variable
training_pro = training[training['Store'] == 1].iloc[:, 1:].drop(['IsHoliday','Type','Size'], axis = 1).reset_index().rename(columns = {'Date' : 'ds', 'Weekly_Sales' : 'y'})



### Modeling - Prophet


# Parameter grid
param_grid = {'seasonality_prior_scale': [5, 10, 20],
              'changepoint_prior_scale': [0.01, 0.05, 0.1],
              'holidays_prior_scale': [5, 10, 20]}
grid = ParameterGrid(param_grid)

#Hyper parameter tuning

rmse_container = []
mape_container = []

for params in grid:
    rmse = []
    mape = []
    for i in range(45):
        
        # Renaming variable
        training_pro = training[training['Store'] == i+1].iloc[:, 1:].drop(['IsHoliday','Type','Size'], axis = 1).reset_index().rename(columns = {'Date' : 'ds', 'Weekly_Sales' : 'y'})
        testing_pro = testing[testing['Store'] == i+1].iloc[:, 1:].drop(['IsHoliday','Type','Size'], axis = 1).reset_index().rename(columns = {'Date' : 'ds'})

        # Splitting training and validation set
        test_days = 30
        training_set = training_pro.iloc[:-test_days, :]
        validation_set = training_pro.iloc[-test_days:, :].drop('y', axis = 1)

        # Creating holiday frame
        holiday_dates = training[(training['Store'] == i+1)&(training.IsHoliday == True)].reset_index().rename(columns = {'Date' : 'ds', 'Weekly_Sales' : 'y'}).ds
        holidays = pd.DataFrame({'holiday' : 'holi',
                                 'ds': pd.to_datetime(holiday_dates),
                                 'lower_window': -3,
                                 'upper_window': 1})
        
        # Building model
        m = Prophet(growth = "linear",
                yearly_seasonality = True,
                weekly_seasonality = True,
                daily_seasonality = False,
                holidays = holidays,
                seasonality_mode = "multiplicative",
                seasonality_prior_scale = params['seasonality_prior_scale'],
                holidays_prior_scale = params['holidays_prior_scale'],
                changepoint_prior_scale = params['changepoint_prior_scale'])
        m.add_regressor('Temperature')
        m.add_regressor('Fuel_Price')
        m.add_regressor('CPI')
        m.add_regressor('Unemployment')
        m.fit(training_set)

        # Cross-validation
        df_cv = cross_validation(m,
                             period='31 days', 
                             horizon = '150 days',
                             initial = '600 days',
                             parallel = "processes")

        # Gathering the rmse & MAPE
        error = np.sqrt(mean_squared_error(df_cv['y'], df_cv['yhat']))
        mape_error = MAPE(df_cv['y'], df_cv['yhat'])
        
        rmse.append(error)
        mape.append(mape_error)
    rmse_container.append(rmse)
    mape_container.append(mape)
    

# Show the minimum average MAPE for 45 Stores
avg_mape_for_all_param = [sum(mape)/45 for mape in mape_container]
print(avg_mape_for_all_param.min())

        


### Ensemble Modeling: Prophet+XGBoost


#Set the parameters
parameters = {'learning_rate': 0.1,
              'max_depth': 3,
              'colsample_bytree': 1,
              'subsample': 1,
              'min_child_weight': 1,
              'gamma': 1,
              'random_state': 1502,
              'eval_metric': "rmse",
              'objective': "reg:squarederror"}

# Fitting best parameter for the model
m = Prophet(growth = "linear",
                yearly_seasonality = True,
                weekly_seasonality = True,
                daily_seasonality = False,
                holidays = holidays,
                seasonality_mode = "multiplicative",
                seasonality_prior_scale = 10,
                holidays_prior_scale = 20,
                changepoint_prior_scale = 0.1)
m.add_regressor('Temperature')
m.add_regressor('Fuel_Price')
m.add_regressor('CPI')
m.add_regressor('Unemployment')
m.fit(training_set)

# Building 45 models for 45 stores
rmse = []
mape = []

for i in range(45):

    training_pro = training[training['Store'] == i+1].iloc[:, 1:].drop(['IsHoliday','Type','Size'], axis = 1).reset_index().rename(columns = {'Date' : 'ds', 'Weekly_Sales' : 'y'})
    testing_pro = testing[testing['Store'] == i+1].iloc[:, 1:].drop(['IsHoliday','Type','Size'], axis = 1).reset_index().rename(columns = {'Date' : 'ds'})

    # Splitting training and validation set for Prophet prediction
    test_days = 30
    training_set = training_pro.iloc[:-test_days, :]
    validation_set = training_pro.iloc[-test_days:, :].drop('y', axis = 1)

    forecast_xgb = m.predict(training_pro.drop('y', axis = 1))

    # Combining the original dataset with the prophet variable
    prophet_variables = forecast_xgb.loc[:, ["trend", "holi", "weekly", "yearly"]]
    df_xgb = pd.concat([training_pro, prophet_variables], axis = 1)

    # Splitting training and test set for XGBoost prediction
    test_days = 30
    training_set = df_xgb.iloc[:-test_days, :]
    validation_set = df_xgb.iloc[-test_days:, :]

    # Isolating X and Y
    y_train = training_set.y
    y_val = validation_set.y
    X_train = training_set.iloc[:, 2:]
    X_val = validation_set.iloc[:, 2:]

    # Creating XGBoost Matrices
    Train = xgb.DMatrix(data = X_train, label = y_train)
    Test = xgb.DMatrix(data = X_val, label = y_val)

    # Building XGBoost Model
    model = xgb.train(params = parameters,
                      dtrain = Train,
                      num_boost_round = 200, #166
                      evals = [(Test, "y")],
                      verbose_eval = 15)

    # Predicting with XGBoost
    predictions_xgb = pd.Series(model.predict(Test), name = "XGBoost")
    predictions_xgb.index = validation_set.ds
    predictions_xgb
    
    # Setting the index
    training_set.index = training_set.ds
    validation_set.index = validation_set.ds

    # Gathering the rmse & MAPE
    error = np.sqrt(mean_squared_error(validation_set['y'], predictions_xgb))
    mape_error = MAPE(validation_set['y'], predictions_xgb)
    rmse.append(error)
    mape.append(mape_error)

# Show the average MAPE for 45 Stores
print(sum(mape)/45)


