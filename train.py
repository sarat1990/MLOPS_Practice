import pickle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#################
def training():
    # 1. Generate Synthetic Data
    # 20 features, 1000 samples
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Feature Scaling
    # Crucial for Logistic Regression convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize and Train Model
    # Using 'lbfgs' solver as it is the default and robust for small-medium datasets
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # 5. Evaluation
    predictions = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, predictions))

    # 6. Persistence
    # We save both the model and the scaler. 
    # Without the scaler, future raw data cannot be correctly processed.
    model_data = {
        "model": model,
        "scaler": scaler
    }

    with open("logistic_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print("Model and scaler saved to 'logistic_model.pkl'")

if __name__ == "__main__":
    training()