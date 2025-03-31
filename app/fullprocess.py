
import os
import json
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import subprocess

# Load config.json
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
source_data_path = os.path.join('sourcedata')

##################Check and read new data
#first, read ingestedfiles.txt
def check_for_new_data():
    # Read ingestedfiles.txt
    try:
        with open(os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt'), 'r') as f:
            ingested_files = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        # If ingestedfiles.txt doesn't exist, assume no files have been ingested
        ingested_files = []

    # Get list of files in source data folder
    source_files = [f for f in os.listdir(os.path.join(os.getcwd(), source_data_path)) if f.endswith('.csv')]

    # Determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    new_files = [f for f in source_files if f not in ingested_files]

    return new_files

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
new_files = check_for_new_data()
if not new_files:
    print("No new data found. Exiting process.")
    exit()

print(f"New data found: {new_files}")

# Copy new files to input folder
for file in new_files:
    source_file = os.path.join(os.getcwd(), source_data_path, file)
    dest_file = os.path.join(os.getcwd(), input_folder_path, file)
    os.system(f'cp {source_file} {dest_file}')

# Ingest new data
ingestion.merge_multiple_dataframe()

# Train model on new data
training.train_model()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
def check_for_model_drift():
    # Get score from the deployed model
    try:
        with open(os.path.join(os.getcwd(), prod_deployment_path, 'latestscore.txt'), 'r') as f:
            deployed_score = float(f.read())
    except FileNotFoundError:
        # If latestscore.txt doesn't exist, assume no model has been deployed
        deployed_score = 0

    # Get score from the model that uses the newest ingested data
    new_score = scoring.score_model()

    print(f"Deployed score: {deployed_score}")
    print(f"New score: {new_score}")

    # Check for model drift (if new score is better than deployed score)
    return new_score > deployed_score

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
model_drift = check_for_model_drift()
if not model_drift:
    print("No model drift detected. Exiting process.")
    exit()

print("Model drift detected. Proceeding with redeployment.")

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
deployment.store_model_into_pickle()

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
diagnostics.model_predictions()
diagnostics.dataframe_summary()
diagnostics.missing_data()
diagnostics.execution_time()
diagnostics.outdated_packages_list()

reporting.score_model()

# Call API endpoints
subprocess.run(['python', 'apicalls.py'])

print("Process completed successfully.")
