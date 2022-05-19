import pandas as pd
import numpy as nm
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as mtp
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_csv("C:/Users/ISHRAT/OneDrive/Desktop/ML/Salary_Data.csv")
x= df.iloc[:, 0].values  
y= df.iloc[:, 1].values   
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0) 
regressor= LinearRegression()  
regressor.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))
y_pred= regressor.predict(x_test.reshape(-1,1))
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)
r2=metrics.r2_score(y_test, y_pred) #R2
print("\nR2{}",r2)
mtp.scatter(x_test, y_test, color="green")    #actual testing data
mtp.plot(x_test, y_pred, color="blue")  #prediction line  
mtp.title("Salary vs Experience (Testing Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()  
