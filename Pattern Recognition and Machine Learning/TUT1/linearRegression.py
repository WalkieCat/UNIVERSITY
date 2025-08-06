import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1200,2,1,1995],
             [1500,3,2,2002],
             [1800,3,2,1985],
             [1350,2,1,1998],
             [2000,4,3,2010]])
y=np.array([250,320,280,300,450])

model = LinearRegression()
model.fit(X, y)

newData = np.array([[1650,3,2,2005],
                    [1400,2,1,2000]])
predictedPrice = model.predict(newData)
print(f"Predicted price: ", predictedPrice)