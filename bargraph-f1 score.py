import pandas as pd
import matplotlib.pyplot as plt

# Data for the algorithms and their performance metrics
data = [
    {"algorithm": "LightGBM", "accuracy": 0.8809, "f1_score": 0.8728, "auc": 0.8809},
    {"algorithm": "XGBoost", "accuracy": 0.8985, "f1_score": 0.8914, "auc": 0.8985},
    {"algorithm": "CatBoost", "accuracy": 0.9319, "f1_score": 0.9276, "auc": 0.9319},
    {"algorithm": "ExtraTrees", "accuracy": 0.8261, "f1_score": 0.8280, "auc": 0.8261},
    {"algorithm": "Decision Trees", "accuracy": 0.6882, "f1_score": 0.6986, "auc": 0.6882},
    {"algorithm": "Random Forest", "accuracy": 0.7354, "f1_score": 0.7433, "auc": 0.7354},
    {"algorithm": "ANN", "accuracy": 0.7002, "f1_score": 0.6868, "auc": 0.7727},
    {"algorithm": "Neural Network", "accuracy": 0.7061, "f1_score": 0.7421, "auc": 0.7369},
    {"algorithm": "Stacking Ens.", "accuracy": 0.9468, "f1_score": 0.9451, "auc": 0.9512}
]

# Creating a DataFrame from the data
df = pd.DataFrame(data)

# Sorting the DataFrame by 'f1_score' in ascending order
df_sorted = df.sort_values(by="f1_score")

# Plotting the F1 Score column in a vertical bar graph
plt.figure(figsize=(12, 6))
bars = plt.bar(df_sorted["algorithm"], df_sorted["f1_score"], color="skyblue")
plt.ylabel("F1 Score")
plt.xlabel("Algorithms")
plt.title("F1 Score of Algorithms")
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding the F1 score values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), ha='center', va='bottom')

# Display the plot
plt.tight_layout()
plt.show()
