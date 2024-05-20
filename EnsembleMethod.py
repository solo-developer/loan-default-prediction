import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

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
categorical_features = ["Education", "EmploymentType", "MaritalStatus", "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]
cat_features_indices = [features.columns.get_loc(col) for col in categorical_features]

# Convert categorical features using OrdinalEncoder
encoder = OrdinalEncoder()
features[categorical_features] = encoder.fit_transform(features[categorical_features])

# Convert encoded features to integer type
features[categorical_features] = features[categorical_features].astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Apply SMOTENC to the training data
smote = SMOTENC(categorical_features=cat_features_indices, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define base models
base_models = [
    ('catboost', CatBoostClassifier(objective= "Logloss",learning_rate=0.1, iterations=100, random_seed=42, eval_metric='AUC', cat_features=categorical_features)),
    ('lightgbm', LGBMClassifier(objective='binary', learning_rate=0.1, n_estimators=100, random_state=42, categorical_feature=cat_features_indices)),
]

# Define the final stacking classifier (meta-learner) with XGBoost
final_estimator = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1, n_estimators=100, random_state=42, 
                                    enable_categorical=True, 
                                    tree_method='hist')

# Create the Stacking Classifier
stacking_ensemble = StackingClassifier(estimators=base_models, final_estimator=final_estimator, passthrough=True)

# Nested cross-validation setup
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Train and evaluate the stacking ensemble with nested CV
for train_index, val_index in skf.split(X_train_resampled, y_train_resampled):
    X_train_inner, X_val_inner = X_train_resampled.iloc[train_index], X_train_resampled.iloc[val_index]
    y_train_inner, y_val_inner = y_train_resampled.iloc[train_index], y_train_resampled.iloc[val_index]

    # Fit the stacking ensemble on the inner fold
    stacking_ensemble.fit(X_train_inner, y_train_inner)
    inner_preds = stacking_ensemble.predict_proba(X_val_inner)[:, 1]

    # Evaluate the model on the inner validation fold
    inner_accuracy = accuracy_score(y_val_inner, inner_preds > 0.5)
    inner_f1 = f1_score(y_val_inner, inner_preds > 0.5)
    inner_auc = roc_auc_score(y_val_inner, inner_preds)

    print(f"Inner Fold Performance: Accuracy: {inner_accuracy:.4f}, F1 Score: {inner_f1:.4f}, AUC-ROC: {inner_auc:.4f}")


# Evaluate the model on the outer test set
stacking_ensemble.fit(X_train_resampled, y_train_resampled)
# test_preds = stacking_ensemble.predict_proba(X_test)[:, 1]

# Make predictions on the validation set
try:
    predictions = stacking_ensemble.predict(X_test)
except Exception as e:
    print(f"Error during prediction: {e}")

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
auc = roc_auc_score(y_test, predictions)

print("Testing Set Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
