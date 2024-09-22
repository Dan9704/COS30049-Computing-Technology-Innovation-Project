import os
import pandas as pd

# Update this to the correct folder path on your Mac
folder_path = '/Users/admin/Documents/Dataset/MelbourneOlympicPark'  # Replace with the actual path to your folder

# Output file name for the combined CSV file
output_csv = os.path.join(folder_path, 'combined_output.csv')

# Function to combine all CSV files based on matching headers
def combine_csv_files(folder_path, output_csv):
    # List all CSV files in the directory and sort by name
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

    # Print found CSV files for debugging
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    print(f"Found CSV files (in order): {csv_files}")
    
    combined_csv = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        print(f"Processing file: {file_path}")  # Debugging statement
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        # Append the DataFrame to the list, matching columns by name
        combined_csv.append(df)

    # Concatenate all DataFrames in the list, aligning by column names
    if combined_csv:
        combined_csv = pd.concat(combined_csv, ignore_index=True, sort=False)
        # Write the combined CSV to a file
        combined_csv.to_csv(output_csv, index=False)
        print(f"Combined CSV saved to {output_csv}")
    else:
        print("No valid CSV files were combined.")

# Execute the CSV combination
combine_csv_files(folder_path, output_csv)
