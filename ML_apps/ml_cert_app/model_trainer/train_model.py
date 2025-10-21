import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

# Loading sample  data
wine_df = pd.read_csv('../../../ref_datasets/winequality-red.csv',sep=';')
wine_df['decision'] = np.where(wine_df['quality'] > 5.5, 1, 0)
X = wine_df[
    [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
]
y = wine_df["decision"]  # original data only has two deicisons 0=Reject, 1=Approve
# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=10
)

# Train a classifier
model = RandomForestClassifier(n_estimators=100, random_state=10)
model.fit(X_train, y_train)

# Evaluate (just to check)
accuracy = model.score(X_test, y_test)
print(f"Model trained successfully! Test Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "certificate_model.pkl")
print("Model saved as certificate_model.pkl")
