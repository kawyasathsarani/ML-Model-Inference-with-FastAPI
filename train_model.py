from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Step 1: Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 4: Evaluate
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Step 5: Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
