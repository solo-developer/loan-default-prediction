import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

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

# Encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])

# Standardize the data
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Apply SMOTENC to the training data
smote = SMOTENC(categorical_features=[data.columns.get_loc(col)-1 for col in categorical_features], random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, target)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping], verbose=2)

# Make predictions on the validation set
predictions = model.predict(X_val)
predictions = (predictions > 0.4).astype(int)

# Evaluate model performance
accuracy = accuracy_score(y_val, predictions)
f1 = f1_score(y_val, predictions)
auc = roc_auc_score(y_val, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
