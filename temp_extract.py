import pandas as pd

# 1. Load the cleaned dataset
# Change this filename if your data is saved under a different name
input_file = 'cleaned_tickets.csv' 
df = pd.read_csv(input_file)

# 2. Filter for 'Low' priority
# (Using .str.capitalize() just in case some are 'low' and others are 'Low')
df['Priority'] = df['Priority'].str.capitalize() 
low_priority_df = df[df['Priority'] == 'Low']

# 3. Specify the exact columns you want to keep
desired_columns = ['Unnamed: 0', 'Body', 'Department', 'Priority', 'Tags', 'clean_text']

# Safety check: Only select columns that actually exist in the dataframe
# (This prevents crashes if 'Unnamed: 0' isn't actually in the file)
columns_to_keep = [col for col in desired_columns if col in low_priority_df.columns]
final_df = low_priority_df[columns_to_keep]

# 4. Save to a new CSV file
output_file = 'low_priority_extracted.csv'
final_df.to_csv(output_file, index=False)

# Print a success message with the count
print(f"✅ Success! Extracted {len(final_df)} 'Low' priority tickets.")
print(f"📁 Saved to: {output_file}")