import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
cat_features_indices = [data.columns.get_loc(col)-1 for col in categorical_features if col in data.columns]

# Calculate class imbalance ratio
class_counts = target.value_counts()
class_imbalance_ratio = class_counts.iloc[1] / class_counts.iloc[0]
print(f"Class imbalance ratio: {class_imbalance_ratio:.2f}")

# Oversampling (recommended for this case)
smote = SMOTENC(categorical_features, random_state=42)
X_train, y_train = smote.fit_resample(features, target)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Preprocessing (One-Hot Encoding)
categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
numerical_transformer = 'passthrough'  # Pass numerical features through without transformation
transformer = ColumnTransformer(transformers=[
  ('cat', categorical_transformer, categorical_features),
  ('num', numerical_transformer, [col for col in X_train.columns if col not in categorical_features])
])

X_train_encoded = transformer.fit_transform(X_train)
X_val_encoded = transformer.transform(X_val)

# Define ANN model (simple example)
model = Sequential()
model.add(Dense(units=32, activation='relu', input_shape=(X_train_encoded.shape[1],)))  # Hidden layer with 32 neurons and ReLU activation
model.add(Dense(units=1, activation='sigmoid'))  # Output layer with 1 neuron and sigmoid activation for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
try:
  model.fit(X_train_encoded, y_train, epochs=15, batch_size=32)  # Adjust epochs and batch_size as needed
except Exception as e:
  print(f"Error during model training: {e}")

# Make predictions on the validation set
try:
  predictions = model.predict(X_val_encoded)
except Exception as e:
  print(f"Error during prediction: {e}")

# Evaluate model performance
try:
  # Check if sigmoid activation is used in the output layer (assuming binary classification)
  if 'sigmoid' in [layer.activation for layer in model.layers]:  # Check for sigmoid activation in any layer

    # Model outputs probabilities, use classification metrics directly
    accuracy = accuracy_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
  else:
    # Model might output raw scores, apply threshold or use alternative metrics
    threshold = 0.4
    binary_predictions = (predictions > threshold).astype(int)
    accuracy = accuracy_score(y_val, binary_predictions)
    f1 = f1_score(y_val, binary_predictions)

    # Additionally, calculate AUC for raw scores (optional)
    auc = roc_auc_score(y_val, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC (for raw scores): {auc:.4f}")
except Exception as e:
  print(f"Error during evaluation: {e}")

print("Model training and evaluation complete!")    
