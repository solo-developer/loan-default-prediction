import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTENC
import time
import matplotlib.pyplot as plt

try:
    # Load data from CSV file (replace with your file path)
    data = pd.read_csv("Loan_default.csv")
except FileNotFoundError:
    print("Error: File 'Loan_default.csv' not found. Please ensure the file exists in the same directory as your script.")
    exit()

# Separate features and target variable
features = data.drop(["Default", "LoanID"], axis=1)
target = data["Default"]

# Define categorical features explicitly (CatBoost requires this)
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

# Calculate class imbalance ratio
class_counts = target.value_counts()
class_imbalance_ratio = class_counts.iloc[1] / class_counts.iloc[0]  # Ratio of non-defaults to defaults
print(f"Class imbalance ratio: {class_imbalance_ratio:.2f}")

# Initialize k-fold cross-validation for base learners
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize arrays to store out-of-fold predictions from base learners
catboost_oof_preds = np.zeros(len(features))
lightgbm_oof_preds = np.zeros(len(features))

# Measure training time for base learners and predictions
start_time = time.time()

# Perform k-fold cross-validation for base learners
for train_index, val_index in skf.split(features, target):
    X_train_fold, X_val_fold = features.iloc[train_index], features.iloc[val_index]
    y_train_fold, y_val_fold = target.iloc[train_index], target.iloc[val_index]

    # Apply SMOTENC to the training fold for base learners
    smote_base = SMOTENC(categorical_features=cat_features_indices, random_state=42)
    X_train_fold_resampled, y_train_fold_resampled = smote_base.fit_resample(X_train_fold, y_train_fold)

    # Train CatBoost model
    catboost_params = {
        "objective": "Logloss",
        "eval_metric": "F1",
        "learning_rate": 0.1,
        "iterations": 1000,
        "random_seed": 42,
        "custom_loss": ["AUC"],
        "verbose": False,
        "cat_features": cat_features_indices
    }
    catboost_model = CatBoostClassifier(**catboost_params)
    catboost_model.fit(X_train_fold_resampled, y_train_fold_resampled, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=10, verbose=False)

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
    lightgbm_model.fit(X_train_fold_resampled, y_train_fold_resampled, eval_set=[(X_val_fold, y_val_fold)])

    # Generate out-of-fold predictions for training set
    catboost_oof_preds[val_index] = catboost_model.predict_proba(X_val_fold)[:, 1]
    lightgbm_oof_preds[val_index] = lightgbm_model.predict_proba(X_val_fold)[:, 1]

# Create new features for meta-model
stacking_train_features = pd.DataFrame({
    "CatBoostPreds": catboost_oof_preds,
    "LightGBMPreds": lightgbm_oof_preds
})

# Apply SMOTENC to the entire training data for meta-learner
smote_meta = SMOTENC(categorical_features=cat_features_indices, random_state=42)
X_train_resampled, y_train_resampled = smote_meta.fit_resample(features, target)

# Split resampled data into training and holdout sets (80:20 ratio)
X_train_meta, X_holdout, y_train_meta, y_holdout = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)

# Train XGBoost meta-model on resampled training predictions
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.1,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 100
}
xgb_model = XGBClassifier(**xgb_params)
xgb_model.fit(stacking_train_features, target)
training_time = time.time() - start_time

# Generate predictions for holdout set for meta learner
catboost_holdout_preds = catboost_model.predict_proba(X_holdout)[:, 1]
lightgbm_holdout_preds = lightgbm_model.predict_proba(X_holdout)[:, 1]
stacking_holdout_features = pd.DataFrame({
    "CatBoostPreds": catboost_holdout_preds,
    "LightGBMPreds": lightgbm_holdout_preds
})

# Measure prediction time
start_time = time.time()
meta_preds = xgb_model.predict(stacking_holdout_features)
prediction_time = time.time() - start_time

# Evaluate meta-model performance on holdout set
meta_accuracy = accuracy_score(y_holdout, meta_preds)
meta_f1 = f1_score(y_holdout, meta_preds)
meta_auc = roc_auc_score(y_holdout, meta_preds)

print("Stacking Ensemble Testing Set Performance:")
print(f"Training time (including base learners): {training_time:.2f} seconds")
print(f"Prediction time: {prediction_time:.2f} seconds")
print(f"Accuracy: {meta_accuracy:.4f}")
print(f"F1 Score: {meta_f1:.4f}")
print(f"AUC-ROC: {meta_auc:.4f}")

# Calculate ROC curve data
fpr, tpr, _ = roc_curve(y_holdout, meta_preds)

# Save metrics and parameters to Excel file
try:
    results = pd.DataFrame({
        'Model': ['Stacking Ensemble'],
        'Accuracy': [meta_accuracy],
        'F1 Score': [meta_f1],
        'AUC': [meta_auc],
        'Training Time (s)': [training_time],
        'Testing Time (s)': [prediction_time]
    })

    with pd.ExcelWriter('StackingEnsemble.xlsx', mode='w') as writer:
        results.to_excel(writer, sheet_name='Metrics', index=False)
        pd.DataFrame({'FPR': fpr, 'TPR': tpr}).to_excel(writer, sheet_name='ROC', index=False)

except Exception as e:
    print(f"Error during data saving: {e}")

# Plot AUC-ROC curve
try:
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % meta_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
except Exception as e:
    print(f"Error during AUC-ROC curve plotting: {e}")

print("Model training and evaluation complete!")
