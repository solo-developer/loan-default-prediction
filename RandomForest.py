import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

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

# Calculate class imbalance ratio
class_counts = target.value_counts()
class_imbalance_ratio = class_counts.iloc[1] / class_counts.iloc[0]
print(f"Class imbalance ratio: {class_imbalance_ratio:.2f}")

# Oversampling (recommended for this case)
cat_features_indices = [features.columns.get_loc(col) for col in categorical_features if col in features.columns]
smote = SMOTENC(categorical_features=cat_features_indices, random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, target)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define Random Forest Classifier parameters
params = {
    "n_estimators": 100,
    "max_depth": 6,
    "random_state": 42,
    "n_jobs": -1,
}

# Train the Random Forest model
try:
    start_train_time = time.time()
    model = RandomForestClassifier(**params)
    
    categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    numerical_features = [col for col in X_train.columns if col not in categorical_features]
    transformer = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', 'passthrough', numerical_features)
    ])
    
    X_train_encoded = transformer.fit_transform(X_train)
    X_val_encoded = transformer.transform(X_val)

    model.fit(X_train_encoded, y_train)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time
    print(f"Training time: {training_time:.2f} seconds")
except Exception as e:
    print(f"Error during model training: {e}")

# Make predictions on the validation set
try:
    start_test_time = time.time()
    predictions = model.predict(X_val_encoded)
    predictions_proba = model.predict_proba(X_val_encoded)[:, 1]
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
        'Model': ['Random Forest'],
        'Accuracy': [accuracy],
        'F1 Score': [f1],
        'AUC': [auc],
        'Training Time (s)': [training_time],
        'Testing Time (s)': [testing_time]
    })

    with pd.ExcelWriter('random_forest.xlsx', mode='w') as writer:
        results.to_excel(writer, sheet_name='Metrics', index=False)
        pd.DataFrame({'FPR': fpr, 'TPR': tpr}).to_excel(writer, sheet_name='ROC', index=False)
        pd.DataFrame({
            'Training Time (s)': [training_time],
            'Testing Time (s)': [testing_time]
        }).to_excel(writer, sheet_name='Time', index=False)

except Exception as e:
    print(f"Error during data saving: {e}")

print("Model training and evaluation complete!")
