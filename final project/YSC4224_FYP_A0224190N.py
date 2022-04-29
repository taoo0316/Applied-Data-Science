#!/usr/bin/env python
# coding: utf-8

# **Importing Libraries and Loading Data Sets**

# We first load the relevant libraries required and the three data sets provided for this final project.

# In[1]:


# import the relevant libraries
import pandas as pd
pd.set_option('display.max_columns',100)
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('darkgrid')

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# load the three data sets
calendar = pd.read_csv("calendar.csv")
calendar


# In[3]:


prices = pd.read_csv("prices.csv")
prices


# In[4]:


# find the average prices for each item using the groupby() function in Python
average_prices_by_item = prices.groupby("item_id").mean()
average_prices_by_item


# In[5]:


train = pd.read_csv("train.csv")
train


# In[6]:


# check whether all stores have a unique ID 
len(train["id"].unique()) == len(train)


# We know that we have 30490 stores, each with its unique ID.

# In[7]:


# check for nan for all three data sets
calendar.isnull().values.any()
prices.isnull().values.any()
train.isnull().values.any()


# There are no NA values present in the three data sets.

# **EDA**

# We first conduct some exploratory data analysis to better understand trends associated with the three data sets we have.

# In[8]:


calendar.hist(figsize=(14,14), xrot=45)
plt.show()


# In[9]:


prices.hist(figsize=(7,7), xrot=45)
plt.show()


# In[10]:


average_prices_by_item.hist(figsize=(7,7), xrot=45)
plt.show()


# In[12]:


calendar.describe()


# In[13]:


prices.describe()


# In[14]:


train.describe()


# In[15]:


train.subcat_id.value_counts()


# In[16]:


train.category_id.value_counts()


# Amongst all the products sold, food and cleaning are the two dominant categories. Beauty products are comparatively fewer in this data set.

# In[17]:


train.store_id.value_counts()


# In[18]:


train.region_id.value_counts()


# The most number of stores are located in the East region. There happen to an equal number of stores located in the West and Central region.

# **Statistical Endogenous Methods**

# As you can see, we are given three data sets in this assignment: calendar, prices and train. In particular, we are looking at the train data set with historical data of the 1920 days prior to the forecast starting point. Based on this, we are to provide a 21-day-forecast of the unit sales at the store, product level. We start with using statistical endogenous methods, followed by incorporating features from the other two data sets.

# *1. Naive Approach*

# We first take a naive approach by using the average across all previous days in our forecast. We will split the data set into two : training (80%) and testing (20%).

# In[19]:


# training set for the naive approach
training  = train.iloc[: , 6:1542]
training


# In[20]:


# test set for the naive approach
testing = train.iloc[: , 1542:]
testing


# Here, we take the average of unit sales for each product in our training set and round it up to the nearest integer.

# In[21]:


# calculate the average sales as a basis for our forecast in the training set
training["forecast"] = round(training.mean(axis=1))
training


# In[22]:


# calculate the rmse for all stores using the naive method
testing_squared_error = (testing.sub(training["forecast"], axis=0))**2
testing["mean squared error"] = testing_squared_error.mean(axis=1)
testing["root mean squared error"] = testing["mean squared error"]**0.5
testing


# In[23]:


# the average rmses across all stores using the naive method
np.mean(testing["root mean squared error"])


# The avergae rmse across all products using the naive method is 1.63.

# *2. Seasonal Naive Method*

# Of course, we can do a little bit better. Instead of simply taking the average across previous time periods, we can take into account seasonal factors as well to possibly generate a better model.

# In[24]:


calendar.tail(50)


# In this case, we observe that the time periods we are forecasting (day 1920 to day 1940) lie exclusively in the month of May. Therefore, it would make sense for us to take into account previous sales volumes in the month of May, if seasonality is indeed a determining factor in our predictions.

# To test this, we set the forecast to the last observed value from the same season (e.g., the same month of the previous year). We are going to test on the last 21 days of data that is available in the test set (day 1899 to day 1919) and calculate the rmse. As seen from the data set below, the data is from the month of April. More specifically, from 10 April 2016 to 30 April 2016. Therefore, we should also look for data from 10 April 2015 to 30 April 2015.

# In[25]:


# investigate the date and time of the last 21 days of available data
calendar.iloc[1898:1919]


# In[26]:


# find the corresponding data of the previous season
calendar.iloc[1533:1554]


# In[27]:


# training set for the seasonal naive approach
training_seasonal = train.iloc[:, 1539:1560]
training_seasonal


# In[28]:


# testing set for the seasonal naive approach
testing_seasonal = train.iloc [:, 1904:1926]
testing_seasonal


# In[29]:


seasonal_squared_difference = (testing_seasonal.subtract(training_seasonal, fill_value =0))**2
seasonal_squared_difference


# In[30]:


# calculate the rmse for each store using the naive seasonal approach
seasonal_squared_difference["mean squared error"] = seasonal_squared_difference.mean(axis=1)
seasonal_squared_difference["root mean squared error"] = seasonal_squared_difference["mean squared error"]**0.5
seasonal_squared_difference


# In[31]:


#calculate the average rmse using the seasonal naive approach 
np.mean(seasonal_squared_difference["root mean squared error"])


# The rmse is in fact a little higher than the naive approach, which suggests factors in seasonality are perhaps not determining in our forecast.

# *3. VAR*
# 
# Next, we are going to use VAR, or Vector autoregression, to predict the future unit sales for each of the stores listed in the data set. A statistical model is autoregressive if it predicts future values based on past values. VAR models generalize the single-variable (univariate) autoregressive model by allowing for multivariate time series.

# To use the VAR model, we need to create a multivariate time series. Specifically, the time series should have index showing the date of the respective sales, which can be accessed from the calendar dataset. The sales can be accessed from the train data set. To ensure there is a one-to-one pairing, values in the "d" column of the calendar data set should match with column names in the train data set. 

# In[32]:


# reshape the train data set using the transpose() function in Python
sales = train.iloc[: , 6:1925].transpose()
sales


# In[33]:


# changing the index to date to create a multivariate time series
sales_time_series = sales.set_index([calendar["date"].head(1919)])
sales_time_series


# Now we have a multivariate time series. The index is the date of the sales, and each column has values representing the unit sales for each store we have in the data set.

# To train the entire data set requires too much computational power. We take the first 500 stores as an example. This will hopefully give us an average rmse value that is representative of our model.

# In[34]:


# selecting the first 500 columns to simplify computation
first_500_sales = sales_time_series.iloc[:, :500]
first_500_sales


# In[35]:


# plot the first 500 sales and visualise the patterns
plt_500 = first_500_sales.plot(colormap = "Dark2", figsize = (14,7),legend=False)
plt_500.set_xlabel("Date")
plt_500.set_ylabel("Unit Sales")

plt.show()


# There are no obvious seasonal patterns to be found in this plot. There is a significant dip towards the end of 2012 and start of 2013, which might be worth looking into.

# In[36]:


#checking stationarity
from statsmodels.tsa.vector_ar.vecm import coint_johansen
max(abs(coint_johansen(first_500_sales,-1,1).eig))


# For a series to be stationary, the eigenvalues of |Φ(L)-1| should be less than 1 in modulus. This multivariate time series fulfils the condition.

# In[37]:


#creating the train and validation set
train_var = first_500_sales[:int(0.8*(len(first_500_sales)))]
valid_var = first_500_sales[int(0.8*(len(first_500_sales))):]

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(endog=train_var)
model_fit = model.fit()

# make prediction on validation
prediction_var = model_fit.forecast(model.endog, steps=len(valid_var))


# In[38]:


#converting predictions to dataframe
pred_var = pd.DataFrame(index=range(0,len(prediction_var)),columns=[first_500_sales.columns])
for j in range(0,500):
    for i in range(0, len(prediction_var)):
       pred_var.iloc[i][j] = prediction_var[i][j]

pred_var = pred_var.astype(int)
pred_var


# In[39]:


# calculate the mean rmse for the VAR model
from numpy import sqrt
from sklearn.metrics import mean_squared_error
np.mean(sqrt(mean_squared_error(pred_var, valid_var)))


# The rmse is still relatively low, but it is in fact a little higher than the naive method. 

# **Machine Learning Models**

# Moving on from statistical methods, we now can incorporate certain features from the other two data sets to build up machine learning models to forecast the unit of sales.

# In particular, the store_id and region_id variables seem particularly relevant. So, let us incorporate these two features first. To do that, we use on-hot encoding to convert the two categorical variables into numeric features.

# In[40]:


# select features and target variable for the machine learning models
x_ml = pd.get_dummies(train.iloc[: , :6].drop(["id","item_id","subcat_id","store_id"],axis=1))
y_ml = training["forecast"]


# In[41]:


x_ml


# In[42]:


y_ml


# Now, we are going to first build three simple machine learning models, predicting the average sales for each of the stores given in the training data set. We are only using the category and region variables for these models. In particular, we are going to look at random forest, XGBoost and SVM. All three are strong machine models that tend to perform well in general.

# In[43]:


# create the train and test sets for the machine learning models
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   x_ml, y_ml, test_size=0.2, random_state=42)


# In[44]:


#RandomForest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=0)
rfmodel = rf.fit(x_train,y_train)


# In[45]:


rf_pred = rfmodel.predict(x_test)
rf_pred


# In[46]:


# rmse for RandomForest
from sklearn.metrics import mean_squared_error
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_rmse


# In[47]:


# XGBoost
import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)


# In[48]:


xg_reg.fit(x_train,y_train)

xg_pred = xg_reg.predict(x_test)

xg_pred


# In[49]:


# rmse for XGBoost
xg_rmse = np.sqrt(mean_squared_error(y_test, xg_pred))
xg_rmse


# In[50]:


# SVM
from sklearn import svm
svm_regr = svm.SVR()


# In[51]:


svm_regr.fit(x_train, y_train)

svm_pred = svm_regr.predict(x_test)

svm_pred


# In[52]:


# rmse for SVM
svm_rmse = np.sqrt(mean_squared_error(y_test, svm_pred))
svm_rmse


# In[53]:


# another error for checking
import numpy as np

def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

print(smape(y_test,rf_pred))
print(smape(y_test,xg_pred))
print(smape(y_test,svm_pred))


# All three models have higher rmse than the baseline predictions. This suggests that the categorical variables do not have great predictive power in predicting the unit sales for each store.

# **Evaluation of model and selection of best**

# After going through the modelling process, we are now ready to evaluate the various models that have built up and select the best amongst these models based on the rmse calculated.

# 1. The naive methods work best in predicting sales. 
# 
# 2. Statistical methods using autoregression perform decently well.
# 
# 3. Machine learning methods perform below expectations. 

# **Recommendations**

# From all these, we can start to provide some recommendations for stores so that the unit sales for each product can be maximised.

# 1. Using previous sales as a guide to prepare the number of products could be a good strategy.
# 2. Seasonal patterns are not a great predictor for sales in this particular case.
# 3. Rather than looking at the average error/RMSEs across all the stores, one can perhaps investigate more closely the model's performance on individual stores/products. This could potentially provide answers on which models to use in specific cases. 
