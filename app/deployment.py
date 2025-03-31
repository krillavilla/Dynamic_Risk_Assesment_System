from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    # Create production deployment directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), prod_deployment_path), exist_ok=True)

    # Copy the latest pickle file
    model_file = os.path.join(os.getcwd(), model_path, 'trainedmodel.pkl')
    prod_model_file = os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl')
    os.system(f'cp {model_file} {prod_model_file}')

    # Copy the latestscore.txt file
    score_file = os.path.join(os.getcwd(), model_path, 'latestscore.txt')
    prod_score_file = os.path.join(os.getcwd(), prod_deployment_path, 'latestscore.txt')
    os.system(f'cp {score_file} {prod_score_file}')

    # Copy the ingestedfiles.txt file
    ingested_files = os.path.join(os.getcwd(), dataset_csv_path, 'ingestedfiles.txt')
    prod_ingested_files = os.path.join(os.getcwd(), prod_deployment_path, 'ingestedfiles.txt')
    os.system(f'cp {ingested_files} {prod_ingested_files}')

if __name__ == '__main__':
    store_model_into_pickle()
