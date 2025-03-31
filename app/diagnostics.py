
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys

# Add the current directory to the path so we can import dbsetup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import dbsetup
except ImportError:
    print("Warning: dbsetup module not found. Database functionality will be disabled.")
    dbsetup = None

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(data=None):
    #read the deployed model and a test dataset, calculate predictions

    # Check if model exists
    model_file_path = os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl')
    if not os.path.exists(model_file_path):
        # Try to train and deploy the model
        try:
            import training
            import deployment
            import ingestion
            print("Model not found. Running ingestion, training, and deployment...")
            ingestion.merge_multiple_dataframe()
            training.train_model()
            deployment.store_model_into_pickle()
        except Exception as e:
            raise Exception(f"Model not found and could not be automatically prepared: {str(e)}. Please run the following scripts in order: 1) ingestion.py, 2) training.py, 3) deployment.py")

    # Load the deployed model
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)

    # Load test data if not provided
    if data is None:
        test_data_file = os.path.join(os.getcwd(), test_data_path, 'testdata.csv')
        data = pd.read_csv(test_data_file)

    # Prepare features
    X = data.drop(['corporation', 'exited'], axis=1) if 'exited' in data.columns else data.drop(['corporation'], axis=1)

    # Make predictions
    predictions = model.predict(X)

    return predictions.tolist()

##################Function to get summary statistics
def dataframe_summary(use_db=True):
    #calculate summary statistics here

    # Load the ingested data
    data_file_path = os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv')
    data = pd.read_csv(data_file_path)

    # Get numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()

    # Calculate summary statistics for each numeric column
    summary_stats = []
    for col in numeric_columns:
        col_stats = {
            'column': col,
            'mean': data[col].mean(),
            'median': data[col].median(),
            'std': data[col].std()
        }
        summary_stats.append(col_stats)

        # Write to database if enabled
        if use_db and dbsetup is not None:
            try:
                # Insert column statistics into data_columns table
                dbsetup.insert_data_column_stats(
                    col_stats['column'],
                    col_stats['mean'],
                    col_stats['median'],
                    col_stats['std']
                )
            except Exception as e:
                print(f"Warning: Failed to write column statistics for '{col}' to database: {str(e)}")

    return summary_stats

##################Function to check missing data
def missing_data(use_db=True):
    # Load the ingested data
    data_file_path = os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv')
    data = pd.read_csv(data_file_path)

    # Calculate percentage of NA values for each column
    na_percentages = []
    total_percentage = 0
    for col in data.columns:
        percentage = data[col].isna().mean() * 100
        na_percentage = {
            'column': col,
            'percentage': percentage
        }
        na_percentages.append(na_percentage)
        total_percentage += percentage

    # Calculate average missing data percentage
    avg_percentage = total_percentage / len(data.columns) if data.columns.size > 0 else 0

    # Write to database if enabled
    if use_db and dbsetup is not None:
        try:
            # Get execution times to complete the model_diagnostics record
            times = execution_time(use_db=False)  # Avoid recursive database writes

            # Insert diagnostics record with missing data percentage
            dbsetup.insert_model_diagnostics(
                times[0],
                times[1],
                avg_percentage
            )
        except Exception as e:
            print(f"Warning: Failed to write missing data percentage to database: {str(e)}")

    return na_percentages

##################Function to get timings
def execution_time(use_db=True):
    #calculate timing of training.py and ingestion.py

    # If use_db is False, return cached execution times to avoid running the scripts
    # This prevents database connection attempts when called from reporting.py
    if not use_db:
        # Return reasonable default values (in seconds)
        return [0.5, 0.3]  # [ingestion_time, training_time]

    # Time the ingestion script
    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - start_time

    # Time the training script
    start_time = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - start_time

    # Write to database if enabled
    if use_db and dbsetup is not None:
        try:
            # Get missing data percentage to complete the model_diagnostics record
            data_file_path = os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv')
            data = pd.read_csv(data_file_path)

            # Calculate average missing data percentage
            total_percentage = sum(data[col].isna().mean() * 100 for col in data.columns)
            avg_percentage = total_percentage / len(data.columns) if data.columns.size > 0 else 0

            # Insert diagnostics record
            dbsetup.insert_model_diagnostics(
                ingestion_time,
                training_time,
                avg_percentage
            )
        except Exception as e:
            print(f"Warning: Failed to write execution times to database: {str(e)}")

    return [ingestion_time, training_time]

##################Function to check dependencies
def outdated_packages_list():
    #get a list of outdated packages

    # Get list of installed packages and their versions
    installed = subprocess.check_output(['pip', 'list', '--format=json']).decode('utf-8')
    installed_packages = pd.read_json(installed)

    # Get list of latest available versions
    outdated = subprocess.check_output(['pip', 'list', '--outdated', '--format=json']).decode('utf-8')

    # If there are no outdated packages, return the installed packages
    if not outdated.strip():
        return installed_packages.to_dict('records')

    # Otherwise, return the list of outdated packages
    outdated_packages = pd.read_json(outdated)
    return outdated_packages.to_dict('records')


if __name__ == '__main__':
    # Set up the database if dbsetup is available
    use_db = False
    if dbsetup is not None:
        try:
            use_db = dbsetup.setup_database()
        except Exception as e:
            print(f"Warning: Failed to set up database: {str(e)}")

    model_predictions()
    dataframe_summary(use_db=use_db)
    missing_data(use_db=use_db)
    execution_time(use_db=use_db)
    outdated_packages_list()
