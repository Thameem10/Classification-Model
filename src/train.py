import joblib
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils import load_data, split_data


# ----------------------------
# Load Data
# ----------------------------
df = load_data("df_file.csv")


# ----------------------------
# Split Data
# ----------------------------
X_train, X_test, y_train, y_test = split_data(df, target_column="Label")

# IMPORTANT: Select only text column
X_train = X_train["Text"]
X_test = X_test["Text"]


# ----------------------------
# Create Pipeline (TF-IDF + Model)
# ----------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )),
    ("clf", MultinomialNB())
])


# ----------------------------
# Stratified K-Fold Cross Validation
# ----------------------------
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=kfold
)

print("Cross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


# ----------------------------
# Train Model
# ----------------------------
pipeline.fit(X_train, y_train)


# ----------------------------
# Evaluate On Test Data
# ----------------------------
y_pred = pipeline.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", acc)

# ----------------------------
# check training accuracy
# ----------------------------
train_pred = pipeline.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

print("Training Accuracy:", train_acc)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ----------------------------
# Save Full Pipeline
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "nb_pipeline.pkl"

joblib.dump(pipeline, model_path)

print("\nModel saved successfully.")