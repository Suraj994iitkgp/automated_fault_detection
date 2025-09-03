import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('fault_data.csv')

# Quick look at the data
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize class distribution if 'fault_label' is your target column
df['fault_label'].value_counts().plot(kind='bar')
plt.title('Fault Label Distribution')
plt.xlabel('Fault Type')
plt.ylabel('Count')
plt.show()
