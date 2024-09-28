import pandas as pd
from scipy.stats import zscore  # For outlier detection
from sklearn.preprocessing import MinMaxScaler  # For normalization

# Step 1: Load the dataset
df = pd.read_csv(r'C:\Users\LENOVO\OneDrive - Swinburne University\innovation\assignment 2\melbourne_olympic_park_combined_file.csv', encoding='latin1', on_bad_lines='skip')

# Step 2: Display initial info and missing values
print("\nInitial Dataset Info:\n")
df.info()
print("\nMissing Values in Each Column:\n")
print(df.isnull().sum())

# Step 3: Remove duplicate rows
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
df = df.drop_duplicates()

# Step 4: Handle Missing Values for numeric columns
numeric_columns = [
    'maximum_gust_kmh', 'wind_spd_kmh', 'maximum_gust_spd',
    'wind_gust_spd', 'gust_kmh', 'wind_spd'
]
for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].fillna(df[column].mean())  # Fill with mean

# Step 5: Handle Missing Values for categorical columns
categorical_columns = ['maximum_gust_dir', 'wind_dir', 'wind_dir_deg']
for column in categorical_columns:
    if column in df.columns:
        df[column] = df[column].fillna(df[column].mode()[0])  # Fill with mode

# Step 6: Convert 'time-local' column to datetime and coerce errors, with `utc=True`
if 'time-local' in df.columns:
    df['time-local'] = pd.to_datetime(df['time-local'], errors='coerce', utc=True)

# Step 7: Drop rows where 'time-local' is NaT
df = df.dropna(subset=['time-local'])

# Step 8: Remove duplicate rows 
df.drop_duplicates(inplace=True)

# Step 9: Remove rows with invalid dates ('1970-01-01')
df = df[df['time-local'] != '1970-01-01']

# Step 10: Convert remaining object columns to numeric where appropriate
additional_numeric_columns = [
    'air_temperature', 'msl_pres', 'minimum_air_temperature', 
    'pres', 'rainfall', 'qnh_pres', 'delta_t', 'rel-humidity',
    'maximum_air_temperature', 'apparent_temp', 'dew_point', 
    'rain_ten', 'rain_hour', 'rainfall_24hr'
]

for column in additional_numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].fillna(df[column].mean())  # Fill with mean for these columns

# Step 11: Handle outliers using Z-scores
if 'air_temperature' in df.columns: 
    df['z_score'] = zscore(df['air_temperature'])
    df = df[df['z_score'].abs() < 3]  # Keep rows with Z-scores within the threshold
    df = df.drop(columns=['z_score'])  # Drop the z-score column after filtering

# Step 12: Normalize numeric columns using Min-Max Scaling
scaler = MinMaxScaler()

# Check which numeric columns are in the dataframe before normalizing
all_numeric_columns = numeric_columns + additional_numeric_columns
all_numeric_columns = [col for col in all_numeric_columns if col in df.columns]
df[all_numeric_columns] = scaler.fit_transform(df[all_numeric_columns])

# Step 13: Final data overview after cleaning and normalization
print("\nCleaned and Normalized Dataset:\n")
print(df.head())
print("\nFinal Dataset Info:\n")
print(df.info())

# Step 14: Save the cleaned and normalized dataset to a new CSV file
df.to_csv(r'C:\Users\LENOVO\OneDrive - Swinburne University\innovation\assignment 2\melbourne_olympic_park_combined_file_cleaned.csv', index=False)
