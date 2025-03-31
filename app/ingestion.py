import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import sys

# Add the current directory to the path so we can import dbsetup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import dbsetup
except ImportError:
    print("Warning: dbsetup module not found. Database functionality will be disabled.")
    dbsetup = None


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe(use_db=True):
    #check for datasets, compile them together, and write to an output file

    # Get list of files in input folder
    filenames = os.listdir(os.path.join(os.getcwd(), input_folder_path))

    # Initialize an empty dataframe to store all data
    df_combined = pd.DataFrame()

    # Record the files that were ingested
    ingested_files = []

    # Read and combine all CSV files
    for file in filenames:
        if file.endswith('.csv'):
            file_path = os.path.join(os.getcwd(), input_folder_path, file)
            df_temp = pd.read_csv(file_path)
            df_combined = pd.concat([df_combined, df_temp], axis=0)
            ingested_files.append(file)

            # Write to database if enabled
            if use_db and dbsetup is not None:
                try:
                    # Insert record into ingested_files table
                    dbsetup.insert_ingested_file(file, len(df_temp))
                except Exception as e:
                    print(f"Warning: Failed to write file '{file}' to database: {str(e)}")

    # Drop duplicates
    df_combined = df_combined.drop_duplicates()

    # Create output folder if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), output_folder_path), exist_ok=True)

    # Write the combined dataframe to finaldata.csv
    output_file_path = os.path.join(os.getcwd(), output_folder_path, 'finaldata.csv')
    df_combined.to_csv(output_file_path, index=False)

    # Write the list of ingested files to ingestedfiles.txt
    with open(os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        for file in ingested_files:
            f.write(f"{file}\n")

    # Write the combined dataframe to the database if enabled
    if use_db and dbsetup is not None:
        try:
            # Insert dataframe into dataset table
            dbsetup.insert_dataframe_to_dataset(df_combined)
        except Exception as e:
            print(f"Warning: Failed to write combined dataframe to database: {str(e)}")

    return df_combined


if __name__ == '__main__':
    # Set up the database if dbsetup is available
    use_db = False
    if dbsetup is not None:
        try:
            use_db = dbsetup.setup_database()
        except Exception as e:
            print(f"Warning: Failed to set up database: {str(e)}")

    merge_multiple_dataframe(use_db=use_db)
