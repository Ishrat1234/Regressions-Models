import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
dataset = pd.read_csv('https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X.reshape(-1,1))
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y.reshape(-1,1))
print("Prediction value of Salary for 5.5 years of experience:" ,pol_reg.predict(poly_reg.fit_transform([[5.5]])))
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X.reshape(-1,1))), color='blue')
    plt.title('Polynomial Regression Model')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return
viz_polymonial()
