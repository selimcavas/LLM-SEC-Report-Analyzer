import os
import pandas as pd

# Directory containing the CSV files
directory = 'data_collection/CSVs'

# List all CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Initialize an empty list to hold the DataFrames
dfs = []

# For each CSV file
for csv_file in csv_files:
    # Extract the ticker symbol from the file name
    ticker = csv_file[:-4]

    # Load the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(directory, csv_file))

    # Rename the column that includes different fields
    df.rename(columns={df.columns[0]: 'Field'}, inplace=True)

    # Add a new column to the DataFrame that contains the ticker symbol
    df['Ticker'] = ticker

    # Make 'Ticker' the first column
    df = df[['Ticker'] + [col for col in df.columns if col != 'Ticker']]

    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined DataFrame to a CSV file
combined_df.to_csv('combined_data.csv', index=False)
