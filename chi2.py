import pandas as pd
from scipy.stats import chi2_contingency

# Load the data
df = pd.read_csv('Electric_Vehicle_Population_Data.csv')

# Calculate cumulative counts for each manufacturer over the years
make_year_counts = df.groupby(['Model Year', 'Make']).size().groupby(level=1).cumsum().reset_index(name='Count')

# Get the top 4 manufacturers by total count
top_makes = make_year_counts.groupby('Make')['Count'].max().nlargest(4).index

# Filter data for top 4 manufacturers
top_make_counts = make_year_counts[make_year_counts['Make'].isin(top_makes)]

# Sum counts over the years for individual manufacturers
target_counts = top_make_counts.groupby('Make')['Count'].max().reset_index(name='Target Count')

# Calculate associations between the target and individual 4 features for top 4 manufacturers
associations = []

for make in top_makes:
    make_data = df[df['Make'] == make]
    total_count = make_data.shape[0]
    
    # Mean Electric Range, neglecting 0 values
    mean_range = make_data[make_data['Electric Range'] != 0]['Electric Range'].mean()
    
    # Chi-squared test for 'Electric Vehicle Type'
    contingency_table_ev_type = pd.crosstab(make_data['Model Year'], make_data['Electric Vehicle Type'])
    chi2_ev_type, p_ev_type, dof_ev_type, expected_ev_type = chi2_contingency(contingency_table_ev_type)
    
    # Chi-squared test for 'CAFV Eligibility'
    contingency_table_cafv = pd.crosstab(make_data['Model Year'], make_data['Clean Alternative Fuel Vehicle (CAFV) Eligibility'])
    chi2_cafv, p_cafv, dof_cafv, expected_cafv = chi2_contingency(contingency_table_cafv)
    
    # Chi-squared test for 'Electric Range'
    contingency_table_range = pd.crosstab(make_data['Model Year'], make_data['Electric Range'])
    chi2_range, p_range, dof_range, expected_range = chi2_contingency(contingency_table_range)
    
    # Chi-squared test for 'Model'
    contingency_table_model = pd.crosstab(make_data['Model Year'], make_data['Model'])
    chi2_model, p_model, dof_model, expected_model = chi2_contingency(contingency_table_model)
    
    associations.append({
        'Make': make,
        'Target Count': total_count,
        'Mean Electric Range': mean_range,
        'Chi2 EV Type': int(chi2_ev_type),
        'P-value EV Type': f"{p_ev_type:.7e}",
        'Chi2 CAFV': int(chi2_cafv),
        'P-value CAFV': f"{p_cafv:.7e}",
        'Chi2 Range': int(chi2_range),
        'P-value Range': f"{p_range:.7e}",
        'Chi2 Model': int(chi2_model),
        'P-value Model': f"{p_model:.7e}"
    })

associations_df = pd.DataFrame(associations)

# Save the results as a CSV file with a different name
associations_df.to_csv('top_4_manufacturers_associations_with_model.csv', index=False)

print("Results have been saved to 'top_4_manufacturers_associations_with_model.csv'")

