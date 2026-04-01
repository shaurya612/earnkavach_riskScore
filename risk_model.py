import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# STEP 1: Load dataset
df = pd.read_csv("data.csv")

# STEP 2: Select features and target
X = df[['rain', 'aqi', 'traffic', 'zone_risk', 'disruptions']]
y = df['risk_score']

# STEP 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# STEP 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 5: Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Model trained successfully!")
print("MSE:", mse)

# STEP 6: Save model
joblib.dump(model, "risk_model.pkl")

print("Model saved as risk_model.pkl")

