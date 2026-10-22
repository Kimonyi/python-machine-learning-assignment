import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data: Years of Experience vs. Salary
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([30000, 35000, 42000, 50000, 58000])

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict salary for 6 years of experience
predicted_salary = model.predict([[6]])
print(f"Supervised ML - Linear Regression Prediction:")
print(f"Predicted salary for 6 years of experience: ${predicted_salary[0]:,.2f}")