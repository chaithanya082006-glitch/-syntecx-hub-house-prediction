import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("Housing.csv")

# Convert yes/no values
data['mainroad'] = data['mainroad'].map({'yes':1,'no':0})
data['guestroom'] = data['guestroom'].map({'yes':1,'no':0})
data['basement'] = data['basement'].map({'yes':1,'no':0})

# Select features
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
# Accuracy
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Example prediction
example = pd.DataFrame([[2000, 3, 2]],
columns=['area', 'bedrooms', 'bathrooms'])

price = model.predict(example)

print("Predicted House Price:", price[0])

# Save model
joblib.dump(model, "house_price_model.pkl")

print("Model Saved Successfully")