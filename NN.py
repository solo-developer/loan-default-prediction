import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time

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

# Convert categorical features to numerical values
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])
    label_encoders[col] = le

# Calculate class imbalance ratio
class_counts = target.value_counts()
class_imbalance_ratio = class_counts.iloc[1] / class_counts.iloc[0]
print(f"Class imbalance ratio: {class_imbalance_ratio:.2f}")

# Oversampling (recommended for this case)
smote = SMOTENC(categorical_features=[features.columns.get_loc(col) for col in categorical_features], random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, target)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Measure training time
start_time = time.time()

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

training_time = time.time() - start_time

# Measure prediction time
start_time = time.time()

# Make predictions on the validation set
predictions_proba = model.predict(X_val)
predictions = (predictions_proba > 0.5).astype("int32")

prediction_time = time.time() - start_time

# Evaluate model performance
accuracy = accuracy_score(y_val, predictions)
f1 = f1_score(y_val, predictions)
auc = roc_auc_score(y_val, predictions_proba)

print(f"Training time: {training_time:.2f} seconds")
print(f"Prediction time: {prediction_time:.2f} seconds")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# Plot AUC-ROC curve
fpr, tpr, _ = roc_curve(y_val, predictions_proba)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Save metrics and parameters to Excel file
try:
    results = pd.DataFrame({
        'Model': ['NeuralNetwork'],
        'Accuracy': [accuracy],
        'F1 Score': [f1],
        'AUC': [auc],
        'Training Time (s)': [training_time],
        'Testing Time (s)': [prediction_time]
    })

    with pd.ExcelWriter('NeuralNetwork.xlsx', mode='w') as writer:
        results.to_excel(writer, sheet_name='Metrics', index=False)
        pd.DataFrame({'FPR': fpr, 'TPR': tpr}).to_excel(writer, sheet_name='ROC', index=False)

except Exception as e:
    print(f"Error during data saving: {e}")

print("Model training and evaluation complete!")
