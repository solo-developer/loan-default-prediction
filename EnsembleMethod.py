import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

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
# Handle categorical features
for col in categorical_features:
  if col in data.columns:
    try:
      features[col] = features[col].astype("category")
    except (KeyError, ValueError) as e:
      print(f"Error converting '{col}' to categorical data type: {e}")

# Calculate class imbalance ratio
class_counts = target.value_counts()
class_imbalance_ratio = class_counts.iloc[1] / class_counts.iloc[0]
print(f"Class imbalance ratio: {class_imbalance_ratio:.2f}")

# Oversampling (recommended for this case)
smote = SMOTENC(categorical_features, random_state=42)
X_train, y_train = smote.fit_resample(features, target)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
numerical_transformer = 'passthrough'
transformer = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features),
    ('num', numerical_transformer, [col for col in data.columns if col not in categorical_features])
])

# Define base models and parameters
base_models = [
  ('catboost', CatBoostClassifier(learning_rate=0.1, iterations=1000, random_seed=42, eval_metric='AUC', cat_features=cat_features_indices)),
 ('lightgbm', LGBMClassifier(objective='binary', learning_rate=0.1, n_estimators=100, random_state=42)),
]

# Define the final stacking classifier (meta-learner)
final_estimator = xgboost.XGBClassifier(objective='binary:logistic', learning_rate=0.1, n_estimators=1000, random_state=42)

stacking_ensemble = StackingClassifier(estimators=base_models, final_estimator=final_estimator, passthrough=True)
stacking_ensemble.fit(X_train, y_train)


# Make predictions on the validation set
predictions = stacking_ensemble.predict(X_val)

# Evaluate model performance
accuracy = accuracy_score(y_val, predictions)
f1 = f1_score(y_val, predictions)
auc = roc_auc_score(y_val, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

print("Model training and evaluation complete!")
