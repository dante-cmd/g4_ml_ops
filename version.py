import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Generate synthetic data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = 2 * X.flatten() + 1 + np.random.normal(0, 0.5, size=10)

# 2. Train the model
model = LinearRegression()
model.fit(X, y)

print(f"Model coefficient: {model.coef_[0]:.2f}")
print(f"Model intercept: {model.intercept_:.2f}")

# 3. Save the model to pickle
output_path = "linear_model.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model, f)
    
print(f"Model saved to {output_path}")

# Optional: Load and verify
with open(output_path, "rb") as f:
    loaded_model = pickle.load(f)
    print(f"Loaded model prediction for X=11: {loaded_model.predict([[11]])[0]:.2f}")
