import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Directory containing the .xlsx files
directory = './'

# Initialize a list to store the AUC values and associated data
roc_data_list = []

# Iterate through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".xlsx"):
        # Load the Excel file
        file_path = os.path.join(directory, filename)
        excel_file = pd.ExcelFile(file_path)
        
        # Check if the 'ROC' sheet exists
        if 'ROC' in excel_file.sheet_names:
            # Read the 'ROC' sheet
            roc_data = pd.read_excel(file_path, sheet_name='ROC')
            
            # Ensure it has the necessary columns
            if 'FPR' in roc_data.columns and 'TPR' in roc_data.columns:
                fpr = roc_data['FPR']
                tpr = roc_data['TPR']
                
                # Calculate the AUC value
                auc_value = auc(fpr, tpr)
                
                # Clean up filename for plot label
                plot_label = filename.replace('.xlsx', '').replace('classifier', '').replace('_', ' ').capitalize()
                
                # Append the data to the list
                roc_data_list.append((plot_label, fpr, tpr, auc_value))

# Sort the data in descending order based on AUC values
roc_data_list.sort(key=lambda x: x[3], reverse=True)

# Initialize plot
plt.figure()

# Plot each ROC curve in sorted order
for plot_label, fpr, tpr, auc_value in roc_data_list:
    plt.plot(fpr, tpr, lw=2, label=f"{plot_label} (AUC = {auc_value:.4f})")

# Plot settings
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Show plot
plt.show()
