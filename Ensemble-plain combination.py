import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, SMOTENC

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

# Split data into 80% training and 20% holdout sets
X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize k-fold cross-validation for base learners
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize arrays to store out-of-fold predictions from base learners
catboost_oof_preds = np.zeros(len(X_train_val))
lightgbm_oof_preds = np.zeros(len(X_train_val))

# Perform k-fold cross-validation for base learners
best_catboost_params = None
best_lightgbm_params = None
best_catboost_auc = 0
best_lightgbm_auc = 0

for train_index, val_index in skf.split(X_train_val, y_train_val):
    X_train_fold, X_val_fold = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
    y_train_fold, y_val_fold = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

    # Apply SMOTENC to the training fold for base learners
    smote_base = SMOTENC(categorical_features=cat_features_indices, random_state=42)
    X_train_fold_resampled, y_train_fold_resampled = smote_base.fit_resample(X_train_fold, y_train_fold)

    # Define parameter grids for CatBoost and LightGBM
    catboost_param_grid = [
        {"learning_rate": 0.1, "iterations": 500, "depth": 4, "l2_leaf_reg": 1},
        {"learning_rate": 0.1, "iterations": 500, "depth": 6, "l2_leaf_reg": 3},
        {"learning_rate": 0.2, "iterations": 1000, "depth": 4, "l2_leaf_reg": 3},
        {"learning_rate": 0.2, "iterations": 1000, "depth": 6, "l2_leaf_reg": 5}
    ]

    lightgbm_param_grid = {
        "learning_rate": [0.1, 0.2],
        "num_leaves": [31, 50],
        "max_depth": [3, 5],
    }

    # Train CatBoost models with each parameter set
    for params in catboost_param_grid:
        params.update({
            "random_seed": 42,
            "custom_loss": ["AUC"],
            "cat_features": cat_features_indices,
            "verbose": False
        })
        catboost_model = CatBoostClassifier(**params)
        catboost_model.fit(X_train_fold_resampled, y_train_fold_resampled, eval_set=(X_val_fold, y_val_fold))
        val_preds = catboost_model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, val_preds)

        if auc > best_catboost_auc:
            best_catboost_auc = auc
            best_catboost_params = params

    # Train LightGBM models with each parameter set
    grid_search = GridSearchCV(estimator=LGBMClassifier(metric='auc', random_seed=42, verbose=-1, force_col_wise=True),
                               param_grid=lightgbm_param_grid, scoring='roc_auc', cv=3, verbose=1)
    grid_search.fit(X_train_fold_resampled, y_train_fold_resampled, eval_set=(X_val_fold, y_val_fold))

    best_lightgbm_params = grid_search.best_params_

    # Generate out-of-fold predictions for training set
    best_catboost_model = CatBoostClassifier(**best_catboost_params)
    best_catboost_model.fit(X_train_fold_resampled, y_train_fold_resampled)
    catboost_oof_preds[val_index] = best_catboost_model.predict_proba(X_val_fold)[:, 1]

    best_lightgbm_model = LGBMClassifier(**best_lightgbm_params)
    best_lightgbm_model.fit(X_train_fold_resampled, y_train_fold_resampled)
    lightgbm_oof_preds[val_index] = best_lightgbm_model.predict_proba(X_val_fold)[:, 1]

print('Out of inner fold')

# Create new features for meta-model
stacking_train_features = pd.DataFrame({
    "CatBoostPreds": catboost_oof_preds,
    "LightGBMPreds": lightgbm_oof_preds
})

# Apply SMOTE to the training data for meta-learner
smote_meta = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote_meta.fit_resample(stacking_train_features, y_train_val)

# Define hyperparameter grid for XGBoost
param_grid_xgb = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Initialize XGBoost classifier
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42, n_estimators=100)

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid_xgb, scoring='roc_auc', cv=3, verbose=1, random_state=42)
random_search.fit(X_train_resampled, y_train_resampled)

# Print best parameters and best score
print("Best parameters found: ", random_search.best_params_)
print("Best AUC-ROC score on CV: {:.4f}".format(random_search.best_score_))

# Train XGBoost meta-model on resampled training predictions with best parameters
best_xgb_params = random_search.best_params_
xgb_model = XGBClassifier(**best_xgb_params, n_estimators=100, random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Retrain best models on the entire training set
smote_final = SMOTENC(categorical_features=cat_features_indices, random_state=42)
X_train_final_resampled, y_train_final_resampled = smote_final.fit_resample(X_train_val, y_train_val)

best_catboost_model = CatBoostClassifier(**best_catboost_params)
best_catboost_model.fit(X_train_final_resampled, y_train_final_resampled)

best_lightgbm_model = LGBMClassifier(**best_lightgbm_params)
best_lightgbm_model.fit(X_train_final_resampled, y_train_final_resampled)

# Generate predictions for holdout set for meta-learner
stacking_holdout_features = pd.DataFrame({
    "CatBoostPreds": best_catboost_model.predict_proba(X_holdout)[:, 1],
    "LightGBMPreds": best_lightgbm_model.predict_proba(X_holdout)[:, 1]
})

# Predict with meta-model on holdout predictions
meta_preds = xgb_model.predict(stacking_holdout_features)

# Evaluate meta-model performance on holdout set
meta_accuracy = accuracy_score(y_holdout, meta_preds)
meta_f1 = f1_score(y_holdout, meta_preds)
meta_auc = roc_auc_score(y_holdout, meta_preds)

print("Stacking Ensemble Testing Set Performance:")
print(f"Accuracy: {meta_accuracy:.4f}")
print(f"F1 Score: {meta_f1:.4f}")
print(f"AUC-ROC: {meta_auc:.4f}")
