import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load your data
data = pd.read_csv("Loan_default.csv")

# Separate features and target variable
features = data.drop(["Default", "LoanID","EmploymentType","MaritalStatus","LoanPurpose","Education","HasMortgage","HasDependents","HasCoSigner"], axis=1)
target = data["Default"]

# Encode categorical features if necessary
categorical_features = features.select_dtypes(include=["object", "category"]).columns.tolist()
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])
    label_encoders[col] = le

# Calculate correlations with target variable (assuming binary classification)
correlations = features.apply(lambda x: x.corr(target))

# Sort correlations in descending order
correlations = correlations.sort_values(ascending=False)

# Print correlations
print("Correlation with Target Variable:")
print(correlations)
