import pandas as pd
from lightgbm import LGBMClassifier
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
features = data.drop(["Default", "LoanID"], axis=1)  # Avoid potentially unnecessary column "LoanID"
target = data["Default"]

# Handle categorical features (if any)
categorical_features = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]
for col in categorical_features:
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

# Alternatively, consider other balancing techniques or cost-sensitive learning within the model.

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define LightGBM parameters
params = {
    "objective": "binary",
    "metric": "auc",  # Emphasize balanced performance due to imbalanced data
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "categorical_feature": categorical_features,
    "early_stopping_rounds": 10,
    "random_seed": 42,
    # **Optional: Class weights for cost-sensitive learning**
    # "class_weight": {0: 1, 1: class_imbalance_ratio}  # Replace 0 with majority class index
}

# Train the LightGBM model with early stopping
try:
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set = [(X_val, y_val)])
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

# Save the model (optional)
#try:
#    model.save_model("loan_default_model.txt")
#except Exception as e:
#    print(f"Error saving model: {e}")

print("Model training and evaluation complete!")
