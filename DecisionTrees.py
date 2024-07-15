import pandas as pd
import time
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
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

# Define categorical features explicitly
categorical_features = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]

# Get the indices of the existing categorical features
cat_features_indices = [features.columns.get_loc(col) for col in categorical_features if col in features.columns]

# Oversampling (recommended for this case)
smote = SMOTENC(categorical_features=cat_features_indices, random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, target)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train[categorical_features])
X_val_encoded = encoder.transform(X_val[categorical_features])

# Convert encoded features to DataFrame
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_features))
X_val_encoded_df = pd.DataFrame(X_val_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Add encoded features back to the original dataframes
X_train_final = pd.concat([X_train.reset_index(drop=True).drop(columns=categorical_features), X_train_encoded_df], axis=1)
X_val_final = pd.concat([X_val.reset_index(drop=True).drop(columns=categorical_features), X_val_encoded_df], axis=1)

# Define Decision Tree Classifier parameters
params = {
    "criterion": "gini",  # Information gain (gini) or entropy
    "max_depth": 6,  # Adjust max_depth based on dataset complexity
    "random_state": 42,
}

# Train the Decision Tree model
try:
    start_train_time = time.time()
    model = DecisionTreeClassifier(**params)
    model.fit(X_train_final, y_train)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time
    print(f"Training time: {training_time:.2f} seconds")
except Exception as e:
    print(f"Error during model training: {e}")

# Make predictions on the validation set
try:
    start_test_time = time.time()
    predictions = model.predict(X_val_final)
    predictions_proba = model.predict_proba(X_val_final)[:, 1]
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
        'Model': ['Decision Tree'],
        'Accuracy': [accuracy],
        'F1 Score': [f1],
        'AUC': [auc],
        'Training Time (s)': [training_time],
        'Testing Time (s)': [testing_time]
    })

    with pd.ExcelWriter('decision_Trees.xlsx', mode='w') as writer:
        results.to_excel(writer, sheet_name='Metrics', index=False)
        pd.DataFrame({'FPR': fpr, 'TPR': tpr}).to_excel(writer, sheet_name='ROC', index=False)
        pd.DataFrame({
            'Training Time (s)': [training_time],
            'Testing Time (s)': [testing_time]
        }).to_excel(writer, sheet_name='Time', index=False)

except Exception as e:
    print(f"Error during data saving: {e}")

print("Model training and evaluation complete!")
