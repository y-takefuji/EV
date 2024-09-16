import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('Electric_Vehicle_Population_Data.csv')

# Count unique 'Make' values over years with 'Model Year'
make_counts = df.groupby(['Model Year', 'Make']).size().reset_index(name='Count')

# Get the top 4 manufacturers by total count
top_makes = make_counts.groupby('Make')['Count'].sum().nlargest(4).index

# Filter data for top 4 manufacturers
top_make_counts = make_counts[make_counts['Make'].isin(top_makes)]

# Save the top 4 sales counts over years to a CSV file
top_make_counts.to_csv('top_make_counts.csv', index=False)

# Define distinct black and white linestyles
linestyles = ['-', '--', '-.', ':']

# Plot trends of top 4 manufacturer counts over years with 4 distinct black and white linestyles
plt.figure(figsize=(12, 6))
for i, make in enumerate(top_makes):
    make_data = top_make_counts[top_make_counts['Make'] == make]
    plt.plot(make_data['Model Year'], make_data['Count'], linestyle=linestyles[i], label=make, color='black')

plt.title('Trends of Top 4 Manufacturer Counts Over Years')
plt.xlabel('Model Year')
plt.ylabel('Count')
plt.legend(title='Make')
plt.grid(True)
plt.tight_layout()
plt.savefig('sales.png', dpi=300)
plt.show()

