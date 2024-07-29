import numpy as np
import seaborn as sns
import matplotlib as plt
import pandas as pd

df = pd.read_csv("geometric_metrics_consolidated.csv")
 
# Create a new column in the dataframe that indicates the range each area falls into
df['Area_Range'] = pd.cut(df['Damage Area'], bins=[0, 20, 40, np.inf], labels=['<20%', '20-40%', '>40%'])
 
# Calculate the counts
counts = df['Area_Range'].value_counts()
 
# Create the histogram
plt.figure(figsize=(10,6))
sns.histplot(df, x='Damage Area', hue='Area_Range', bins=50, palette='viridis', multiple="stack")
 
# Draw vertical lines at 0.2 and 0.4
plt.axvline(20, color='r', linestyle='--')
plt.axvline(40, color='b', linestyle='--')
 
# Add text to indicate the lines
plt.text(20.5, 175,'20%',rotation=0, color='r', fontsize=12)
plt.text(40.5, 175,'40%',rotation=0, color='b', fontsize=12)
 
# Set title and labels
plt.title('Distribution of Area', fontsize=14)
plt.xlabel('Percentage of Total Area', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
 
# Create the legend
plt.legend(title='Area Range', labels=[f'<20% (n={counts["<20%"]})', f'20-40% (n={counts["20-40%"]})', f'>40% (n={counts[">40%"]})'])
 
# Show the plot
plt.show()