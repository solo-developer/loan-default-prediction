import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Plot correlations (horizontal bar chart)
plt.figure(figsize=(10, 6))
bars = plt.barh(correlations.index, correlations.values, color='skyblue')
plt.xlabel('Correlation with Target Variable')
plt.ylabel('Feature')
plt.title('Correlation of Features with Target Variable')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add correlation values as labels on the bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}',
             va='center', ha='left', color='black', fontsize=10)

plt.tight_layout()
plt.show()
