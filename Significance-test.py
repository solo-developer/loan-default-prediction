import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load your dataset (replace with your actual data loading code)
try:
    data = pd.read_csv("Loan_default.csv")
except FileNotFoundError:
    print("Error: File 'Loan_default.csv' not found. Please ensure the file exists in the same directory as your script.")
    exit()

# Assuming 'Default' is your target variable
target_variable = 'Default'

# List of categorical variables you want to test
categorical_variables = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]

# Calculate chi-square statistic and p-value for each categorical variable
chi2_results = []
for var in categorical_variables:
    contingency_table = pd.crosstab(data[var], data[target_variable])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    chi2_results.append((var, chi2, p))

# Sort variables by chi-square statistic (higher chi-square indicates more significant association)
chi2_results.sort(key=lambda x: x[1], reverse=True)

# Create a DataFrame from the results
chi2_df = pd.DataFrame(chi2_results, columns=['Variable', 'Chi-Squared', 'p-Value'])

# Plot the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(chi2_df['Variable'], chi2_df['Chi-Squared'], color='skyblue')
plt.ylabel('Chi-Squared Statistic')
plt.xlabel('Categorical Variable')
plt.title('Chi-Squared Test Results for Categorical Variables')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add chi-squared values as labels on the bars
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}',
             va='bottom', ha='center', color='black', fontsize=10)

plt.tight_layout()
plt.show()
