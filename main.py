import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([4, 5, 6, 3, 5, 7, 3, 9, 13, 14])
plt.scatter(x,y)
plt.savefig("scatterplot.png")
model = LinearRegression()
model.fit(x,y)
print(model.coef_, model.intercept_)
plt.plot(x,model.coef_ * x + model.intercept_, "r")
plt.savefig("regressionline.png")