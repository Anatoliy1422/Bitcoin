# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:21:10 2023

@author: Anatoliy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

# Set the R environment variables
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.0"
os.environ["PATH"] = r"C:\Program Files\R\R-4.3.0\bin\x64" + ";" + os.environ["PATH"]

# Import required R packages and functions
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# Import necessary R packages
r_utils = importr('utils')
r_splines = importr('splines')
r_smooth_spline = robjects.r['smooth.spline']

# Import necessary Python packages
from sklearn.metrics import r2_score

# Read the Bitcoin data from the specified CSV file
df = pd.read_csv('C:/Users/Anatoliy/Downloads/Bitcoin.csv')

# Split the data into a training and test set using 80% as training data
df_train = df.sample(frac=0.8).rename(columns={'Date':'date', 'Close':'price'})[['date', 'price']]
df_test = df.drop(df_train.index).rename(columns={'Date':'date', 'Close':'price'})[['date', 'price']]

# Print the shape (rows and columns) of the training and test sets
print('Train dataset has {} rows and {} columns'.format(df_train.shape[0], df_train.shape[1]))
print('Test dataset has {} rows and {} columns'.format(df_test.shape[0], df_test.shape[1]))

# Perform descriptive statistics on the training and test sets
print(df_train.describe())
print(df_test.describe())

# Create a copy of the training data with non-null price values
df_train_2 = df_train[~df_train.price.isnull()].copy()

# Convert the 'date' column to datetime format
df_train_2.date = pd.to_datetime(df_train_2.date, format='%Y-%m-%d')

# Extract month information from the 'date' column
df_train_2['month'] = df_train_2.date.dt.month
df_train_2['month_name'] = df_train_2.date.dt.month_name()

# Extract weekday information from the 'date' column
df_train_2['weekday'] = df_train_2.date.dt.weekday
df_train_2['weekday_name'] = df_train_2.date.dt.day_name()

# Print the head of the modified training data
print(df_train_2.head())

# Plot the average Bitcoin price per month
ax = df_train_2.groupby(['month', 'month_name']) \
    .agg({'price': np.mean}).reset_index() \
    .sort_values(by=['month']).plot.bar(x='month_name', y='price', color='skyblue', title='Average Bitcoin price per month')
ax.set_xlabel('Month')
ax.set_ylabel('Average Price')
ax.legend().set_visible(False)

# Plot the average Bitcoin price per weekday
ax = df_train_2.groupby(['weekday', 'weekday_name']) \
    .agg({'price': np.mean}).reset_index() \
    .sort_values(by=['weekday']).plot.bar(x='weekday_name', y='price', color='lightgreen', title='Average Bitcoin price per weekday')
ax.set_xlabel('Weekday')
ax.set_ylabel('Average Price')
ax.legend().set_visible(False)

# Create a new DataFrame with the average price per day for the training data
df_train_summary = df_train_2.groupby(['date']) \
    .agg({'price': np.mean}).reset_index()

# Compute the minimum date in the training data
min_date = df_train_summary.date.min()

# Convert the 'date' column to represent days since the minimum date
df_train_summary.date = df_train_summary.date - min_date
df_train_summary.date = df_train_summary.date.dt.days

# Prepare the test data for analysis
df_test_2 = df_test[~df_test.price.isnull()].copy()
df_test_2.date = pd.to_datetime(df_test_2.date, format='%Y-%m-%d')
df_test_2['month'] = df_test_2.date.dt.month
df_test_2['month_name'] = df_test_2.date.dt.month_name()
df_test_2['weekday'] = df_test_2.date.dt.weekday
df_test_2['weekday_name'] = df_test_2.date.dt.day_name()

# Create a new DataFrame with the average price per day for the test data
df_test_summary = df_test_2.groupby(['date']) \
    .agg({'price': np.mean}).reset_index()

# Convert the 'date' column in the test data to represent days since the minimum date
df_test_summary.date = df_test_summary.date - min_date
df_test_summary.date = df_test_summary.date.dt.days

# Print a sample of the test data summary
print(df_test_summary.sample(frac=0.01))

# Access R's 'predict' and 'lm' functions
r_predict = robjects.r["predict"]
r_lm = robjects.r["lm"]

# Create an empty DataFrame to store R2 scores
df_r2_scores = pd.DataFrame({'R2': []})

# Function that fits a polynomial model on the training dataset, plots the regression line, and reports the R2 score on the test set
def fit_polynomial(order, df_train_summary=df_train_summary, df_test_summary=df_test_summary, title='model'):
    r_date_train = robjects.FloatVector(df_train_summary.date)
    r_price_train = robjects.FloatVector(df_train_summary.price)
    crypto_train_r = robjects.DataFrame({'date': r_date_train, 'price': r_price_train})

    # Create the formula for the polynomial model using R's 'poly' function
    simple_formula = robjects.Formula("price ~ poly(date, {}, raw=TRUE)".format(order))

    # Fit the linear model using R's 'lm' function
    crypto_lm = r_lm(formula=simple_formula, data=crypto_train_r)

    # Create a DataFrame for prediction with the training dates
    predict_df = robjects.DataFrame({'date': robjects.FloatVector(df_train_summary.date)})

    # Make predictions using R's 'predict' function
    predictions = r_predict(crypto_lm, predict_df)

    # Create subplots for visualization
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the training data points and the regression line
    df_train_summary.plot.scatter(x='date', y='price', c='Red', title="{} - Bitcoin train data".format(title), ax=ax[0])
    ax[0].set_xlabel("date")
    ax[0].set_ylabel("price")
    ax[0].plot(predict_df.rx2("date"), predictions)

    # Create a DataFrame for prediction with the test dates
    predict_test_df = robjects.DataFrame({'date': robjects.FloatVector(df_test_summary.date)})

    # Make predictions on the test set using R's 'predict' function
    predictions_test = r_predict(crypto_lm, predict_test_df)

    # Plot the test data points and the regression line
    df_test_summary.plot.scatter(x='date', y='price', c='Green', title="{} - Bitcoin test data".format(title), ax=ax[1])
    ax[1].set_xlabel("date")
    ax[1].set_ylabel("price")
    ax[1].plot(predict_test_df.rx2("date"), predictions_test)

    # Calculate the R2 score on the test set
    r2 = r2_score(df_test_summary.price, predictions_test)

    # Print the model title and the R2 score on the test set
    print(title)
    print("R2 on test: {}".format(r2))

    # Add the R2 score to the DataFrame
    df_r2_scores.loc[title] = r2
print(fit_polynomial(5, title='5-order Polynomial'))
print(fit_polynomial(25, title='25-order Polynomial'))

# function that fits a cubic B-spline on training dataset, plots the regression line and report the R2 on test
# The 'knots' parameter can be specified as a list of knots
def fit_b_spline(knots=np.quantile(df_train_summary.date,[.25,.5,.75]), df_train_summary=df_train_summary, df_test_summary=df_test_summary, title='model'):
    # Convert train data to R vectors
    r_date_train = robjects.FloatVector(df_train_summary.date) 
    r_price_train = robjects.FloatVector(df_train_summary.price)

    # Convert knots to R vector
    r_quarts = robjects.FloatVector(knots)
    
    # Define the B-spline formula with knots and polynomial terms
    bs_formula = robjects.Formula("price ~ bs(date, knots=r_quarts) + bs(date**2, knots=r_quarts) + bs(date**3, knots=r_quarts)")
    bs_formula.environment['price'] = r_price_train
    bs_formula.environment['date'] = r_date_train
    bs_formula.environment['r_quarts'] = r_quarts

    # Fit the B-spline model
    bs_model = r_lm(bs_formula)
    
    # Prepare prediction data for training set
    predict_df = robjects.DataFrame({'date': robjects.FloatVector(df_train_summary.date)})
    
    # Generate predictions using the trained model
    bs_out = r_predict(bs_model, predict_df)

    # Plot the training set scatter plot and B-spline regression line
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    df_train_summary.plot.scatter(x='date',y='price',c='Red',title="{} - Litecoin train data".format(title), ax=ax[0])
    ax[0].set_xlabel("date")
    ax[0].set_ylabel("price");
    ax[0].plot(predict_df.rx2("date"),bs_out);
    
    # Prepare prediction data for test set
    predict_test_df = robjects.DataFrame({'date': robjects.FloatVector(df_test_summary.date)})
    
    # Generate predictions for the test set using the trained model
    bs_out_test = r_predict(bs_model, predict_test_df)
    
    # Plot the test set scatter plot and B-spline regression line
    df_test_summary.plot.scatter(x='date',y='price',c='Green',title="{} - Litecoin test data".format(title), ax=ax[1])
    ax[1].set_xlabel("date")
    ax[1].set_ylabel("price");
    ax[1].plot(predict_test_df.rx2("date"),bs_out_test);
    
    # Calculate R-squared on the test set
    r2 = r2_score(df_test_summary.price, bs_out_test)
    print(title)    
    print("R2 on test: {}".format(r2))
    
    # Add R-squared value to the DataFrame
    df_r2_scores.loc[title] = r2
# function that fits a cubic Natural-spline on training dataset, plots the regression line and report the R2 on test
# the degree of freedom is calculated via CV
def fit_n_spline(df_train_summary=df_train_summary, df_test_summary=df_test_summary, title='model'):
    # Convert train data to R vectors
    r_date_train = robjects.FloatVector(df_train_summary.date) 
    r_price_train = robjects.FloatVector(df_train_summary.price)
    
    # Perform cross-validated smoothing spline fit
    spline_cv = r_smooth_spline(x=r_date_train, y=r_price_train, cv=True, tol=1/1000000)    
    df_cv = int(spline_cv.rx2("df")[0])

    # Generate the natural spline design matrix
    ns_design = r_splines.ns(r_date_train, df=df_cv)
    
    # Define the natural spline formula with polynomial terms
    ns_formula = robjects.Formula("price ~ bs(date) + bs(date**2) + bs(date**3)")
    ns_formula.environment['price'] = r_price_train
    ns_formula.environment['date'] = r_date_train

    # Fit the natural spline model
    ns_model = r_lm(ns_formula)
    
    # Prepare prediction data for training set
    predict_df = robjects.DataFrame({'date': robjects.FloatVector(df_train_summary.date)})
    
    # Generate predictions using the trained model
    ns_out = r_predict(ns_model, predict_df)

    # Plot the training set scatter plot and natural spline regression line
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    df_train_summary.plot.scatter(x='date',y='price',c='Red',title="{} - Litecoin train data".format(title), ax=ax[0])
    ax[0].set_xlabel("date")
    ax[0].set_ylabel("price");
    ax[0].plot(predict_df.rx2("date"),ns_out);
    
    # Prepare prediction data for test set
    predict_test_df = robjects.DataFrame({'date': robjects.FloatVector(df_test_summary.date)})
    
    # Generate predictions for the test set using the trained model
    ns_out_test = r_predict(ns_model, predict_test_df)
    
    # Plot the test set scatter plot and natural spline regression line
    df_test_summary.plot.scatter(x='date',y='price',c='Green',title="{} - Litecoin test data".format(title), ax=ax[1])
    ax[1].set_xlabel("date")
    ax[1].set_ylabel("price");
    ax[1].plot(predict_test_df.rx2("date"),ns_out_test);
    
    # Calculate R-squared on the test set
    r2 = r2_score(df_test_summary.price, ns_out_test)
    print(title + " - degree of freedom via cross-validation: " + str(df_cv))
    print("R2 on test: {}".format(r2))
    
    # Add R-squared value to the DataFrame
    df_r2_scores.loc[title] = r2
# function that fits a smoother spline on training dataset, plots the regression line and report the R2 on test
# the penalty lambda is calculated via CV
def fit_smoothing_spline(df_train_summary=df_train_summary, df_test_summary=df_test_summary, title='model'):
    # Convert train data to R vectors
    r_date_train = robjects.FloatVector(df_train_summary.date) 
    r_price_train = robjects.FloatVector(df_train_summary.price)
    
    # Perform cross-validated smoothing spline fit
    spline_cv = r_smooth_spline(x=r_date_train, y=r_price_train, cv=True, tol=1/1000000)    
    lambda_cv = spline_cv.rx2("lambda")[0]

    # Plot the training set scatter plot and smoothing spline fit
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    df_train_summary.plot.scatter(x='date',y='price',c='Red',title="{} - Litecoin train data".format(title), ax=ax[0])
    ax[0].set_xlabel("date")
    ax[0].set_ylabel("price");
    ax[0].plot(spline_cv.rx2("x"),spline_cv.rx2("y"));
    
    # Prepare prediction data for the test set
    predict_test_df = robjects.DataFrame({'date': robjects.FloatVector(df_test_summary.date)})
    
    # Generate predictions using the smoothing spline fit
    predictions = r_predict(spline_cv, predict_test_df)
    
    # Plot the test set scatter plot and predicted values
    df_test_summary.plot.scatter(x='date',y='price',c='Green',title="{} - Litecoin test data".format(title), ax=ax[1])
    ax[1].set_xlabel("date")
    ax[1].set_ylabel("price");
    ax[1].plot(df_test_summary.date,np.array(predictions.rx2("y")).ravel());
    
    # Calculate R-squared on the test set
    r2 = r2_score(df_test_summary.price, np.array(predictions.rx2("y")).ravel())
    print(title + " - lambda found via cross-validation: " + str(lambda_cv))
    print("R2 on test: {}".format(r2))
    
    # Add R-squared value to the DataFrame
    df_r2_scores.loc[title] = r2
    
    # Return the smoothing spline fit object
    return spline_cv
# Fit a smoothing spline on the training data and obtain the spline fit object
spline_cv = fit_smoothing_spline(title='Smoothing spline')

# Create a list of numbers ranging from 3129 to 3129+48 (exclusive)
lol = list(range(3129, 3129+48))

# Print the 'date' column of the df_test_summary DataFrame
print(df_test_summary.date)
# Create a new variable called future_days by appending a new series to the 'date' column of df_test_summary
# The new series contains a range of numbers from 2860 to 2860+5 (exclusive)
future_days = pd.concat([df_test_summary.date, pd.Series(list(range(2860, 2860+5)))], ignore_index=True)
def predict_smoothing_spline(model=spline_cv, future_days=future_days):
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    
    # Create a dataframe for predicting future prices using the specified 'future_days'
    predict_test_df = robjects.DataFrame({'date': robjects.FloatVector(future_days)})
    
    # Use the 'r_predict' function to make predictions using the given model and predict_test_df
    predictions = r_predict(model, predict_test_df)
    
    # Plot the scatter plot of the test data in green color
    df_test_summary.plot.scatter(x='date',y='price',c='Green',title="Bitcoin future price", ax=ax)
    
    # Set the x-axis label to 'date' and y-axis label to 'price'
    ax.set_xlabel("date")
    ax.set_ylabel("price");
    
    # Plot the predicted values for the future days
    ax.plot(future_days, np.array(predictions.rx2("y")).ravel())

def fit_linear_regression(df_train_summary, df_test_summary, title='Linear Regression'):
    # Perform linear regression
    model = LinearRegression()
    model.fit(df_train_summary[['date']], df_train_summary['price'])

    # Generate predictions on training data
    train_predictions = model.predict(df_train_summary[['date']])

    # Generate predictions on test data
    test_predictions = model.predict(df_test_summary[['date']])

    # Calculate R-squared on training data
    train_r2 = r2_score(df_train_summary['price'], train_predictions)

    # Calculate R-squared on test data
    test_r2 = r2_score(df_test_summary['price'], test_predictions)

    # Calculate residuals
    train_residuals = df_train_summary['price'] - train_predictions

    # Calculate standard error of predictions
    std_error = np.std(train_residuals)

    # Calculate confidence interval
    confidence_interval = 1.96 * std_error  # 95% confidence interval

    # Plotting the results
    plt.figure(figsize=(15, 5))
    plt.plot(df_train_summary['date'], df_train_summary['price'], label='Actual')
    plt.plot(df_train_summary['date'], train_predictions, label='Predicted')
    plt.fill_between(df_train_summary['date'], train_predictions - confidence_interval,
                     train_predictions + confidence_interval, color='gray', alpha=0.3,
                     label='Confidence Interval')
    plt.title(title)
    plt.xlabel('date')
    plt.ylabel('price')
    plt.legend()
    plt.show()

    print(title)
    print("R-squared on training data: ", train_r2)
    print("R-squared on test data: ", test_r2)
fit_linear_regression(df_train_summary,df_test_summary)
# Call the function with the provided dataframes df_train_summary and df_test_summary
print(predict_smoothing_spline())