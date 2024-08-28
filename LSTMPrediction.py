import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc, precision_score, recall_score
import matplotlib.pyplot as plt
import keras_tuner as kt

# Load the dataset
file_path = 'processed_dataset1.csv'
data = pd.read_csv(file_path)

data['Default'] = data['Default'].map({'Default': 0, 'Non-Default': 1})

# Separate features and target variable
X = data.drop('Default', axis=1)
y = data['Default']

# Balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Standardize numerical features
numerical_features = X.select_dtypes(include=['number']).columns
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Reshape for LSTM
X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the model-building function for KerasTuner
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=128, step=16), return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=16)))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Initialize the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='my_dir',
    project_name='lstm_tuning'
)

# Perform the hyperparameter search
tuner.search(X_train_reshaped, y_train, epochs=5, validation_data=(X_test_reshaped, y_test), batch_size=32)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the best hyperparameters
print(f"""
The optimal number of units in the first LSTM layer is {best_hps.get('units_1')}.
The optimal number of units in the second LSTM layer is {best_hps.get('units_2')}.
The optimal dropout rate for the first LSTM layer is {best_hps.get('dropout_1')}.
The optimal dropout rate for the second LSTM layer is {best_hps.get('dropout_2')}.
The optimal learning rate is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train_reshaped, y_train, epochs=15, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Predictions
y_pred_prob = model.predict(X_test_reshaped)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate metrics
roc_auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Print metrics
print(f"ROC AUC Score: {roc_auc}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Save metrics to file
output_file = 'LSTM_Results.txt'
with open(output_file, 'w') as file:
    file.write(f"ROC AUC Score: {roc_auc}\n")
    file.write(f"F1 Score: {f1}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
