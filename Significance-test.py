import pandas as pd
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

# Sort variables by p-value (smaller p-value indicates more significant association)
chi2_results.sort(key=lambda x: x[2])

# Print variables in descending order of significance (lower p-value is more significant)
print("Variables and their Chi-square test results (in descending order of significance):")
for var, chi2, p in chi2_results:
    print(f"{var}: Chi-square = {chi2:.4f}, p-value = {p:.4f}")
