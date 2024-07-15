import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import time

def f1_m(y_true, y_pred):
    y_true = K.cast(y_true, dtype=K.floatx())
    y_pred = K.cast(y_pred, dtype=K.floatx())
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

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
cat_features_indices = [data.columns.get_loc(col) - 1 for col in categorical_features if col in data.columns]

# Calculate class imbalance ratio
class_counts = target.value_counts()
class_imbalance_ratio = class_counts.iloc[1] / class_counts.iloc[0]
print(f"Class imbalance ratio: {class_imbalance_ratio:.2f}")

# Oversampling (recommended for this case)
smote = SMOTENC(categorical_features=cat_features_indices, random_state=42)
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

# Compile the model with custom F1 score metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])

# Measure training time
start_time = time.time()

# Train the model
try:
    model.fit(X_train_encoded, y_train, epochs=15, batch_size=32)  # Adjust epochs and batch_size as needed
except Exception as e:
    print(f"Error during model training: {e}")

training_time = time.time() - start_time

# Measure prediction time
start_time = time.time()

# Make predictions on the validation set
try:
    predictions = model.predict(X_val_encoded).ravel()
    binary_predictions = (predictions > 0.4).astype(int)  # Apply threshold to get binary predictions
except Exception as e:
    print(f"Error during prediction: {e}")

prediction_time = time.time() - start_time

# Evaluate model performance using F1 Score
try:
    accuracy = accuracy_score(y_val, binary_predictions)
    f1 = f1_score(y_val, binary_predictions)
    auc = roc_auc_score(y_val, predictions)
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Prediction time: {prediction_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")

# Plot AUC-ROC curve
try:
    fpr, tpr, _ = roc_curve(y_val, predictions)
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
except Exception as e:
    print(f"Error during AUC-ROC curve plotting: {e}")

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
