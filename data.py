import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the original dataset
data = pd.read_csv('Electric_Vehicle_Population_Data.csv')

print(f"Original dataset shape: {data.shape}")
print("Original columns:", data.columns.tolist())

# Remove the first five columns and the last four columns
all_columns = data.columns.tolist()
columns_to_keep = all_columns[5:-5]  # Skip first 5 and last 5
data_filtered = data[columns_to_keep]

print("\nFiltered dataset:")
print(f"Shape: {data_filtered.shape}")
print("Remaining columns:", data_filtered.columns.tolist())

# Create a comprehensive sales data by make and year
def create_comprehensive_sales_data(data):
    # Ensure 'Model Year' and 'Make' are available
    if 'Model Year' not in data.columns or 'Make' not in data.columns:
        raise ValueError("Required columns 'Model Year' or 'Make' were removed. They are needed for grouping.")

    # Basic counts by make and year
    sales_counts = data.groupby(['Model Year', 'Make']).size().reset_index(name='sales')
    
    # Identify numeric and categorical columns
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in data.columns if col not in numeric_cols]
    
    print(f"\nNumeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Aggregate numeric columns
    numeric_aggs = {}
    for col in numeric_cols:
        if col != 'Model Year':  # Skip the groupby column
            numeric_aggs[col] = ['mean', 'median', 'min', 'max']
    
    # Perform aggregation if we have numeric columns to aggregate
    if numeric_aggs:
        sales_data = data.groupby(['Model Year', 'Make']).agg(numeric_aggs).reset_index()
        
        # Flatten the column MultiIndex
        sales_data.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in sales_data.columns]
    else:
        sales_data = sales_counts.copy()
        sales_data = sales_data.drop('sales', axis=1)  # Will add back later
    
    # Add categorical most common values 
    for col in categorical_cols:
        if col not in ['Model Year', 'Make']:
            # Get the most common value for each make and year
            most_common = data.groupby(['Model Year', 'Make'])[col].agg(
                lambda x: x.value_counts().index[0] if not x.empty else 'Unknown'
            ).reset_index(name=f'{col}_most_common')
            
            # Merge with the main sales data
            sales_data = sales_data.merge(most_common, on=['Model Year', 'Make'])
    
    # Add the basic sales count (target variable)
    sales_data = sales_data.merge(sales_counts, on=['Model Year', 'Make'])
    
    # Calculate additional metrics
    if 'Electric Vehicle Type' in data.columns:
        # Calculate percentage of BEVs vs PHEVs
        ev_types = data['Electric Vehicle Type'].unique()
        vehicle_types = pd.crosstab(
            [data['Model Year'], data['Make']], 
            data['Electric Vehicle Type']
        ).reset_index()
        
        # Only calculate BEV percentage if we have both types
        bev_col = 'Battery Electric Vehicle (BEV)'
        phev_col = 'Plug-in Hybrid Electric Vehicle (PHEV)'
        
        if bev_col in vehicle_types.columns and phev_col in vehicle_types.columns:
            vehicle_types['BEV_pct'] = vehicle_types[bev_col] / (
                vehicle_types[bev_col] + vehicle_types[phev_col]
            ) * 100
            sales_data = sales_data.merge(vehicle_types[['Model Year', 'Make', 'BEV_pct']], 
                                          on=['Model Year', 'Make'], 
                                          how='left')
    
    return sales_data

# Create the comprehensive sales data
try:
    comprehensive_sales_data = create_comprehensive_sales_data(data_filtered)

    # Save to CSV
    comprehensive_sales_data.to_csv('ev_sales_by_make_year_comprehensive.csv', index=False)
    print("\nCreated ev_sales_by_make_year_comprehensive.csv with sales and additional features")

    # Display the first few rows and shape of the comprehensive data
    print("\nComprehensive sales data (first few rows):")
    print(comprehensive_sales_data.head())
    print(f"Shape of comprehensive sales data: {comprehensive_sales_data.shape}")
    
except Exception as e:
    print(f"Error creating comprehensive data: {e}")
