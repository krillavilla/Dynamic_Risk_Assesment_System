from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

# Add the current directory to the path so we can import dbsetup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import dbsetup
except ImportError:
    print("Warning: dbsetup module not found. Database functionality will be disabled.")
    dbsetup = None


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for model scoring
def score_model(use_db=True):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # Create model directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), model_path), exist_ok=True)

    # Load the trained model
    model_file_path = os.path.join(os.getcwd(), model_path, 'trainedmodel.pkl')
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)

    # Load test data
    test_data_file = os.path.join(os.getcwd(), test_data_path, 'testdata.csv')
    test_data = pd.read_csv(test_data_file)

    # Separate features and target
    X_test = test_data.drop(['corporation', 'exited'], axis=1)
    y_test = test_data['exited']

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate F1 score
    f1 = metrics.f1_score(y_test, y_pred)

    # Write the result to latestscore.txt
    score_file_path = os.path.join(os.getcwd(), model_path, 'latestscore.txt')
    with open(score_file_path, 'w') as f:
        f.write(str(f1))

    # Write the score to the database if enabled
    if use_db and dbsetup is not None:
        try:
            # Insert record into model_scores table
            dbsetup.insert_model_score('trainedmodel', f1)
        except Exception as e:
            print(f"Warning: Failed to write model score to database: {str(e)}")

    return f1

if __name__ == '__main__':
    # Set up the database if dbsetup is available
    use_db = False
    if dbsetup is not None:
        try:
            use_db = dbsetup.setup_database()
        except Exception as e:
            print(f"Warning: Failed to set up database: {str(e)}")

    score_model(use_db=use_db)
