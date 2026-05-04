# =====================================
# HEART DISEASE PREDICTION SYSTEM
# Using 3 ML Algorithms
# =====================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =====================================
# STEP 1: Load Dataset
# =====================================

data = pd.read_csv("processed.cleveland.data", header=None)

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

data.columns = columns

# =====================================
# STEP 2: Data Preprocessing
# =====================================

data.replace("?", np.nan, inplace=True)
data = data.apply(pd.to_numeric)

# Fill missing values with median
data.fillna(data.median(), inplace=True)

# Convert target to binary
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

X = data.drop("target", axis=1)
y = data["target"]

# =====================================
# STEP 3: Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# STEP 4: Feature Scaling
# =====================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# STEP 5: Define Models
# =====================================

log_model = LogisticRegression(max_iter=1000)
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)

models = {
    "Logistic Regression": log_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model
}

# =====================================
# STEP 6: Train & Evaluate Models
# =====================================

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=================================")
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print()