import pandas as pd
import time
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
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
cat_features_indices = [data.columns.get_loc(col)-1 for col in categorical_features if col in data.columns]

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
smote = SMOTENC(categorical_features=cat_features_indices, random_state=42)
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
    start_train_time = time.time()
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time
    print(f"Training time: {training_time:.2f} seconds")
except Exception as e:
    print(f"Error during model training: {e}")

# Make predictions on the validation set
try:
    start_test_time = time.time()
    predictions = model.predict(X_val)
    predictions_proba = model.predict_proba(X_val)[:, 1]
    end_test_time = time.time()
    testing_time = end_test_time - start_test_time
    print(f"Testing time: {testing_time:.2f} seconds")
except Exception as e:
    print(f"Error during prediction: {e}")

# Evaluate model performance
try:
    accuracy = accuracy_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    auc = roc_auc_score(y_val, predictions_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")

# Plot AUC-ROC curve
try:
    fpr, tpr, thresholds = roc_curve(y_val, predictions_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
except Exception as e:
    print(f"Error during AUC-ROC curve plotting: {e}")

# Save metrics and parameters to Excel file
try:
    results = pd.DataFrame({
        'Model': ['CatBoostClassifier'],
        'Accuracy': [accuracy],
        'F1 Score': [f1],
        'AUC': [auc],
        'Training Time (s)': [training_time],
        'Testing Time (s)': [testing_time]
    })

    with pd.ExcelWriter('CatBoostClassifier.xlsx', mode='w') as writer:
        results.to_excel(writer, sheet_name='Metrics', index=False)
        pd.DataFrame({'FPR': fpr, 'TPR': tpr}).to_excel(writer, sheet_name='ROC', index=False)

except Exception as e:
    print(f"Error during data saving: {e}")

print("Model training and evaluation complete!")
