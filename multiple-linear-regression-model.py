import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets
house = datasets.load_boston()
train_x, test_x, train_y, test_y = train_test_split(house.data,
                                                    house.target,
                                                    test_size=0.2,
                                                    random_state=42)
lr = LinearRegression()
lr.fit(train_x, train_y) 
pred_y = lr.predict(test_x)
df = pd.DataFrame({'Actual': test_y.flatten(), 'Predicted': pred_y.flatten()})
print(df)
r2=metrics.r2_score(test_y, pred_y)
print("\nR2{}",r2)
