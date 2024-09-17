import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('Final_Report_of_the_Asian_American_Quality_of_Life__AAQoL_.csv')

# Drop 'Survey ID' and 'Qualtiy of Life' columns
df = df.drop(columns=['Survey ID', 'Qualtiy of Life'])

# Define the correct age bins
age_bins = [18, 34, 50, 66, 82, 98]
age_labels = ['18-34', '35-50', '51-66', '67-82', '83-98']
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

# Calculate the ratios of unique 'Present Mental Health' values per age group
mental_health_counts = df.groupby('Age Group')['Present Mental Health'].value_counts(normalize=True).unstack().fillna(0)

# Reorder the columns to be in the desired order
desired_order = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
mental_health_counts = mental_health_counts[desired_order]

# Define more distinct patterns including black fill
patterns = ['///','//', '\\\\', '||', ' ', '+', 'x', 'o', 'O', '.', '*', 'black']

# Plot the graph
fig, ax = plt.subplots(figsize=(10, 7))
bottom = np.zeros(len(mental_health_counts))

for i, column in enumerate(mental_health_counts.columns):
    color = 'black' if patterns[i % len(patterns)] == 'black' else 'white'
    ax.bar(mental_health_counts.index, mental_health_counts[column], bottom=bottom, label=column, hatch=patterns[i % len(patterns)], color=color, edgecolor='black')
    bottom += mental_health_counts[column]

ax.set_ylabel('Ratio')
ax.set_title('Ratios of Unique Present Mental Health Values per Age Group')

# Move the legend outside of the plot and zoom out to show more pattern areas
ax.legend(title='Present Mental Health', bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)

# Move the X-axis label outside of the plot
ax.set_xlabel('Age Group', labelpad=20)
plt.tight_layout()
plt.savefig('mental.png',dpi=300)
plt.show()

