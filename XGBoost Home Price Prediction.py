import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load the dataset
df = pd.read_csv('kc_house_data.csv')

# Display the first few rows and check columns
print(df.head())
print(df.columns)

# Display summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Select features and target variable
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
            'lat', 'long', 'sqft_living15', 'sqft_lot15']

X = df[features]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the XGBoost Regressor
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.1, 0.3, 0.5],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5,
                           scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example of making predictions for new data
new_data = pd.DataFrame({
    'bedrooms': [3],
    'bathrooms': [2],
    'sqft_living': [2000],
    'sqft_lot': [5000],
    'floors': [2],
    'waterfront': [0],
    'view': [0],
    'condition': [3],
    'grade': [8],
    'sqft_above': [1600],
    'sqft_basement': [400],
    'yr_built': [1990],
    'yr_renovated': [0],
    'zipcode': [98001],
    'lat': [47.32],
    'long': [-122.21],
    'sqft_living15': [2100],
    'sqft_lot15': [5200]
})

# Standardize the new data
new_data_scaled = scaler.transform(new_data)

# Predict with the best model
predicted_price = best_model.predict(new_data_scaled)
print(f'Predicted Home Price: ${predicted_price[0]:,.2f}')

