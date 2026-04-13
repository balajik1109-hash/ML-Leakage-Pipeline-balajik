# House Price Prediction Project

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Task 1: Create Dataset + Train Model + Predictions-

# STEP 1: Create synthetic dataset:
np.random.seed(42)
n = 60  # number of records
area_sqft = np.random.randint(500, 3000, n)
num_bedrooms = np.random.randint(1, 5, n)
age_years = np.random.randint(0, 30, n)

# Target variable (price in lakhs):
price_lakhs = (
    area_sqft * 0.05 +
    num_bedrooms * 10 -
    age_years * 0.3 +
    np.random.normal(0, 5, n)  # noise
)

# Create DataFrame
df = pd.DataFrame({
    'area_sqft': area_sqft,
    'num_bedrooms': num_bedrooms,
    'age_years': age_years,
    'price_lakhs': price_lakhs
})

print("Dataset Preview:\n")
print(df.head())

# STEP 2: Prepare features and target:
X = df[['area_sqft', 'num_bedrooms', 'age_years']]
y = df['price_lakhs']

# STEP 4: Train model
model = LinearRegression()
model.fit(X, y)

# STEP 3: Print intercept and coefficients:
print("\nIntercept:", model.intercept_)

print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# STEP 4: Predictions:
y_pred = model.predict(X)

print("\nFirst 5 Actual vs Predicted:")
for i in range(5):
    print(f"Actual: {y.iloc[i]:.2f} | Predicted: {y_pred[i]:.2f}")

# Task 2: Evaluation Metrics-

mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("\nModel Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Explanation:
# MAE → average difference between actual and predicted values
# RMSE → prioritizes correcting large errors over small ones
# R² → tells how well the model explains variance (closer to 1 is better)

# Task 3: Residual Analysis & Plot-

# STEP 1: Residual Analysis:
residuals = y - y_pred

# STEP 2: Plotting Histogram:
plt.hist(residuals, bins=10)
plt.title("Residual Distribution")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()

# Explanation:
# Residual = Actual - Predicted
# If Histogram(graph) is centered around 0 → model is good has low-bias and performs well