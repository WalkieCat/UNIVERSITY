import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])

# Reshape X into column vector
X = X.reshape(-1, 1)

degree = 5
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

X_plot = np.linspace(0, 6, 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.scatter(X, y, label='Original Data')
plt.plot(X_plot, y_plot, color='red', label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Fit of Degree {}'.format(degree))
plt.legend()
plt.show()