import pandas as pd

# 1. Load your main cleaned dataset
input_file = 'cleaned_tickets.csv'
df = pd.read_csv(input_file)

# Ensure consistent capitalization just in case
df['Priority'] = df['Priority'].str.capitalize()

# 2. Separate the tickets into three distinct buckets
df_high = df[df['Priority'] == 'High']
df_medium = df[df['Priority'] == 'Medium']
df_low = df[df['Priority'] == 'Low']

print(f"Original counts:\nHigh: {len(df_high)}\nMedium: {len(df_medium)}\nLow: {len(df_low)}\n")

# 3. Calculate how many extra 'Low' tickets we need to match 'Medium'
target_count = len(df_medium)
current_low_count = len(df_low)
needed_low_tickets = target_count - current_low_count

print(f"Duplicating {needed_low_tickets} 'Low' priority tickets to balance the dataset...")

# 4. Randomly duplicate the existing 'Low' tickets to make up the difference
# replace=True is what allows pandas to pick the same ticket multiple times
df_low_extra = df_low.sample(n=needed_low_tickets, replace=True, random_state=42)

# 5. Combine the original dataset with the new extra 'Low' tickets
df_balanced = pd.concat([df, df_low_extra])

# 6. Shuffle the entire dataset (so the duplicates aren't all clustered at the bottom)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 7. Save to a new CSV file
output_file = 'cleaned_tickets_balanced.csv'
df_balanced.to_csv(output_file, index=False)

print(f"\n✅ Success! Saved to {output_file}")
print("New balanced counts:")
print(df_balanced['Priority'].value_counts())