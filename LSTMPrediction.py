import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Load data from CSV file (replace with your file path)
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

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])
    label_encoders[col] = le

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# Oversampling using SMOTENC
smote = SMOTENC(categorical_features=[features.columns.get_loc(col) for col in categorical_features], random_state=42)
X_resampled, y_resampled = smote.fit_resample(features_scaled, target)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Reshape input for RNN (samples, timesteps, features)
X_train = np.expand_dims(X_train, axis=1)
X_val = np.expand_dims(X_val, axis=1)

# Build RNN model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with 'AUC' and 'accuracy' metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model on the validation set
val_loss, val_auc, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

# Make predictions
y_pred_proba = model.predict(X_val)
y_pred = (y_pred_proba > 0.5).astype("int32")

# Calculate additional metrics
val_f1 = f1_score(y_val, y_pred)

# Print and record results in Excel
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")
print(f"Validation AUC: {val_auc:.4f}")

# Record results in Excel
results = {
    "Validation Accuracy": [val_accuracy],
    "Validation F1 Score": [val_f1],
    "Validation AUC": [val_auc]
}

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Define the excel file name
excel_file = "LSTM_Model_Results.xlsx"

# Check if file exists
if not os.path.exists(excel_file):
    # Create a new workbook and add the results
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='LSTM', index=False)
else:
    # Append the results to the existing file
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
        results_df.to_excel(writer, sheet_name='LSTM', index=False)

# Plot AUC-ROC curve
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % val_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print("Model training, evaluation, and recording complete!")
