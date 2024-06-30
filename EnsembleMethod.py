import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder

# Load data from CSV file
try:
    data = pd.read_csv("Loan_default.csv")
except FileNotFoundError:
    print("Error: File 'Loan_default.csv' not found. Please ensure the file exists in the same directory as your script.")
    exit()

# Separate features and target variable
features = data.drop(["Default", "LoanID"], axis=1)
target = data["Default"]

# Define categorical features explicitly
categorical_features = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner"
]

# Get the indices of the existing categorical features
cat_features_indices = [features.columns.get_loc(col) for col in categorical_features if col in features.columns]

# Handle categorical features
for col in categorical_features:
    if col in features.columns:
        try:
            features[col] = features[col].astype("category")
        except (KeyError, ValueError) as e:
            print(f"Error converting '{col}' to categorical data type: {e}")

# Apply SMOTE to the training data
smote = SMOTENC(categorical_features, random_state=42)
X_train, y_train = smote.fit_resample(features, target)

# Split data into training and validation sets
X_train_resampled, X_test, y_train_resampled, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train CatBoost model
catboost_params = {
    "objective": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.1,
    "iterations": 1000,
    "random_seed": 42,
    "verbose": False,
    "cat_features": cat_features_indices
}
catboost_model = CatBoostClassifier(**catboost_params)
catboost_model.fit(X_train_resampled, y_train_resampled, eval_set=[(X_test, y_test)], early_stopping_rounds=10)

# Train LightGBM model
lightgbm_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "random_seed": 42
}
lightgbm_model = LGBMClassifier(**lightgbm_params)
lightgbm_model.fit(X_train_resampled, y_train_resampled, eval_set=[(X_test, y_test)])

# Generate predictions for stacking
catboost_preds_train = catboost_model.predict(X_train_resampled)
lightgbm_preds_train = lightgbm_model.predict(X_train_resampled)

# Create new features for meta-model
stacking_train_features = pd.DataFrame({
    "CatBoostPreds": catboost_preds_train,
    "LightGBMPreds": lightgbm_preds_train
})

# Train XGBoost meta-model
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 100
}
xgb_model = XGBClassifier(**xgb_params)
xgb_model.fit(stacking_train_features, y_train_resampled)

# Generate predictions for test set
catboost_preds_test = catboost_model.predict(X_test)
lightgbm_preds_test = lightgbm_model.predict(X_test)

stacking_test_features = pd.DataFrame({
    "CatBoostPreds": catboost_preds_test,
    "LightGBMPreds": lightgbm_preds_test
})

# Predict with meta-model
meta_preds = xgb_model.predict(stacking_test_features)

# Evaluate meta-model performance
meta_accuracy = accuracy_score(y_test, meta_preds)
meta_f1 = f1_score(y_test, meta_preds)
meta_auc = roc_auc_score(y_test, meta_preds)

print("Stacking Ensemble Testing Set Performance:")
print(f"Accuracy: {meta_accuracy:.4f}")
print(f"F1 Score: {meta_f1:.4f}")
print(f"AUC-ROC: {meta_auc:.4f}")
