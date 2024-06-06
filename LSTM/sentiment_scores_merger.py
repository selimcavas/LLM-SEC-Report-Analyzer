import pandas as pd

# Read the Excel files
df1 = pd.read_excel('sentiment_scores.xlsx')
df2 = pd.read_excel('sentiment_scores_10K.xlsx')

# Concatenate the DataFrames
df = pd.concat([df1, df2])

# Sort by 'ticker' and 'report_name'
df = df.sort_values(['ticker', 'report_name'], ascending=[True, False])

# Write to a new Excel file
df.to_excel('merged_sentiment_scores.xlsx', index=False)