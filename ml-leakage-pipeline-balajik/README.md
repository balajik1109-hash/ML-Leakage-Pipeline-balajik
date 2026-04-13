#  House Price Prediction using Multiple Linear Regression

##  Overview
This project builds a Multiple Linear Regression model to predict house prices based on key property features. A synthetic dataset is generated and used to train and evaluate the model.


##  Dataset Description
The dataset is synthetically created using Python and contains 60 records with the following features:

- **area_sqft** → Area of the house in square feet  
- **num_bedrooms** → Number of bedrooms  
- **age_years** → Age of the property in years  
- **price_lakhs** → Target variable (house price in lakhs)


##  Methodology

### 1. Data Generation
- Used NumPy to create random but realistic housing data
- Added noise to simulate real-world variation

### 2. Model Building
- Applied Multiple Linear Regression using Scikit-learn
- Trained model using:
  - area_sqft
  - num_bedrooms
  - age_years

### 3. Predictions
- Generated predicted house prices
- Compared actual vs predicted values


##  Model Evaluation

The model is evaluated using the following metrics:

- **MAE (Mean Absolute Error)**  
  Measures the average absolute difference between actual and predicted values.

- **RMSE (Root Mean Squared Error)**  
  Penalize larger errors more strongly than MAE by squaring the difference between predicted and actual values. 
- **R² Score**  
  Indicates how well the model explains the variance in the data. Values closer to 1 indicate better performance.


##  Residual Analysis

- Residuals are calculated as:
Residual = Actual - Predicted

- A histogram of residuals is plotted to analyze model performance.

## Interpretation:
- If residuals are centered around zero → model has low bias  
- If distribution is symmetric → model predictions are balanced  
- If roughly bell-shaped → model assumptions are satisfied  

##  Technologies Used

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

##  How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ml-leakage-pipeline-balaji.git
cd ml-leakage-pipeline-balaji