import pandas as pd

df = pd.read_csv('cleaned_tickets_balanced.csv')
print(df['Priority'].value_counts())