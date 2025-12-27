
import pickle
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model(output_dir="artifacts", output_file="logistic_model.pkl"):
    # 1. Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Logistic Regression model
    model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate model
    predictions = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, predictions))

    # 6. Save model and scaler
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_file)
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    print(f"Model and scaler saved to '{out_path}'")

if __name__ == "__main__":
    train_and_save_model()

