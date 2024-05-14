import pandas as pd

# Read the only_lstm_model_evaluations.csv and model_evaluations.csv files
vanilla_lstm_data = pd.read_csv('only_lstm_model_evaluations.csv')
filtered_model_data = pd.read_csv('filtered_model_evaluations.csv')
model_data = pd.read_csv('model_evaluations.csv')

# Drop the 'ticker' column from both DataFrames
vanilla_lstm_data = vanilla_lstm_data.drop('ticker', axis=1)
filtered_model_data = filtered_model_data.drop('ticker', axis=1)
model_data = model_data.drop('ticker', axis=1)

# Filter out negative R2 values
vanilla_lstm_data = vanilla_lstm_data[vanilla_lstm_data['r2'] > 0]
filtered_model_data = filtered_model_data[filtered_model_data['r2'] > 0]
model_data = model_data[model_data['r2'] > 0]

# Calculate the average of the columns in each DataFrame
vanilla_lstm_avg = vanilla_lstm_data.mean()
filtered_model_avg = filtered_model_data.mean()
model_avg = model_data.mean()

# Combine the averages into a single DataFrame
combined_avg = pd.concat(
    [vanilla_lstm_avg, filtered_model_avg, model_avg], axis=1)
combined_avg.columns = ['lstm_avg',
                        'f_sentiment_lstm_avg', 'sentiment_lstm_avg']

# write the combined averages to a new CSV file

combined_avg.to_csv('combined_avg_metrics.csv', index=False)
