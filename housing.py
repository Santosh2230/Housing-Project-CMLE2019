import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extracting the information from csv file
df = pd.read_csv('Housing.txt', delim_whitespace=True, header = -1)
df
# Defining header names
# CRIM      per capita crime rate by town
# ZN        proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS     proportion of non-retail business acres per town
# CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# NOX       nitric oxides concentration (parts per 10 million)
# RM        average number of rooms per dwelling
# AGE       proportion of owner-occupied units built prior to 1940
# DIS       weighted distances to five Boston employment centres
# RAD       index of accessibility to radial highways
# TAX       full-value property-tax rate per 10,000
# PTRATIO   pupil-teacher ratio by town
# B         1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT     percentlower status of the population
# MEDV      Median value of owner-occupied homes in 1000's
col_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',  'B', 'LSTAT', 'MEDV']
df.columns = col_name

# Printing the header contents
df.head()

# Data exploration-mean, median, min, max, etc...
df.describe()

# Filling NaN
df['LSTAT'] = df['LSTAT'].fillna(df['LSTAT'].median())
df['MEDV'] = df['MEDV'].fillna(df['MEDV'].median())
df.info()
# Printing the new data coloumns
print (df['LSTAT'])
print (df['MEDV'])

# Visualisation of selected data
col_assignment = ['PTRATIO', 'B', 'LSTAT', 'MEDV']
sns.pairplot(df[col_assignment])
print (plt.show ())

# Visualization of data in 2D. Heat map is a colored visual summary of information
pd.options.display.float_format = '{:,.2f}'.format
df.corr()
plt.figure(figsize = (16,10))
# Print total data
sns.heatmap(df.corr(), annot = True, cmap = 'seismic')
plt.show()
# Print selected data
col_feature = ['INDUS', 'ZN', 'NOX', 'AGE', 'DIS', 'LSTAT']
sns.heatmap(df[col_feature].corr(), annot = True, cmap = 'seismic')
plt.show()

# Perform linear regression of the data

# Taking the data into X and y variables. 'RM' data contains a single feature
X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values
# Importing LinearRegression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
print(model.coef_)
print(model.intercept_)
# Plotting the model
plt.figure(figsize=(12, 8))
sns.regplot(X, y)
plt.xlabel('Avearge Rooms per dwelling')
plt.ylabel('Median value')
plt.grid(True)
plt.show()

# Actual and predicted data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
AP = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
AP

# Plotting the joint data sets

# RM vs MEDV plot
plt.figure(figsize = (15,8))
sns.regplot(X, y)
sns.jointplot(x = 'RM', y= 'MEDV', kind = 'reg', data = df)
plt.xlabel('Avearge Rooms per dwelling')
plt.ylabel('Median value')
plt.grid(True)
plt.show()
# LSTAT vs MEDV plot
plt.figure(figsize = (15,8))
sns.jointplot(x = 'LSTAT', y= 'MEDV', kind = 'reg', data = df)
plt.xlabel('%Lower status of the population')
plt.ylabel('Median value')
plt.grid(True)
plt.show()




