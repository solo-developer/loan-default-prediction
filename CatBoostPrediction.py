import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
cat_features_indices = [data.columns.get_loc(col)-1 for col in categorical_features if col in data.columns]
print(cat_features_indices)
# Handle categorical features
for col in categorical_features:
    if col in data.columns:
        try:
            features[col] = features[col].astype("category")
        except (KeyError, ValueError) as e:
            print(f"Error converting '{col}' to categorical data type: {e}")

# Calculate class imbalance ratio
class_counts = target.value_counts()
class_imbalance_ratio = class_counts.iloc[1] / class_counts.iloc[0]  # Ratio of non-defaults to defaults
print(f"Class imbalance ratio: {class_imbalance_ratio:.2f}")

# Oversampling (recommended for this case)
smote = SMOTENC(categorical_features, random_state=42)
X_train, y_train = smote.fit_resample(features, target)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define CatBoost parameters
params = {
    "objective": "Logloss",  # Suitable for binary classification
    "eval_metric": "AUC",  # Emphasize balanced performance
    "learning_rate": 0.1,
    "iterations": 1000,  # Adjust based on dataset size and model complexity
    "random_seed": 42,
    "custom_loss": ["AUC"],  # Directly optimize for AUC
    "verbose": False,  # Control model output
    "cat_features": cat_features_indices  # Specify categorical feature indices
}

# Train the CatBoost model with early stopping
try:
    model = CatBoostClassifier(**params)
    print('after model initialisation')
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
except Exception as e:
    print(f"Error during model training: {e}")

# Make predictions on the validation set
try:
    predictions = model.predict(X_val)
except Exception as e:
    print(f"Error during prediction: {e}")

# Evaluate model performance
try:
    accuracy = accuracy_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    auc = roc_auc_score(y_val, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")

print("Model training and evaluation complete!")
