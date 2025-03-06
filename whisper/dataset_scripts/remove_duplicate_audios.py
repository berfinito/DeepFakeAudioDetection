import pandas as pd
import os

# check_duplicate scripts create a duplicates.csv instead of removing
# this script uses that csv to remove the duplicates

duplicates_csv = "duplicates.csv"

def delete_original_files(csv_path):
    df = pd.read_csv(csv_path)

    # Check if 'Original Path' column exists
    if 'Original Path' not in df.columns:
        print("Error: 'Original Path' column not found in the CSV.")
        return

    # Iterate through the 'Original Path' column and delete files
    for file_path in df['Original Path']:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

delete_original_files(duplicates_csv)

print("Deletion process completed.")
