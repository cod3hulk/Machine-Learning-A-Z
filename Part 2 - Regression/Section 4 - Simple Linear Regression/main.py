import pandas as pd

# import dataset
dataset = pd.read_csv("Salary_Data.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# split test and train data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=0)

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the test results
y_pred = regressor.predict(x_test)

import matplotlib.pyplot as plt

# visualizing the training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Slaray vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
