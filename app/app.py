from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import diagnostics
import scoring
import training
import deployment

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    try:
        # Get data from request
        data = request.get_json()
        file_path = data['file_path']

        # Load data
        data = pd.read_csv(file_path)

        # Make predictions
        predictions = diagnostics.model_predictions(data)

        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    try:
        # Check if latestscore.txt exists
        score_file_path = os.path.join(os.getcwd(), prod_deployment_path, 'latestscore.txt')
        if not os.path.exists(score_file_path):
            # Try to generate the score and deploy it
            try:
                import scoring
                import deployment
                print("Score file not found. Running scoring and deployment...")
                scoring.score_model()
                deployment.store_model_into_pickle()
            except Exception as e:
                raise Exception(f"Score file not found and could not be automatically generated: {str(e)}. Please run the following scripts in order: 1) scoring.py, 2) deployment.py")

        # Get F1 score
        with open(score_file_path, 'r') as f:
            score = float(f.read())

        return jsonify({"score": score})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    try:
        # Get summary statistics
        summary_stats = diagnostics.dataframe_summary()

        return jsonify({"summary_statistics": summary_stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics_endpoint():        
    #check timing and percent NA values
    try:
        # Get execution time
        execution_times = diagnostics.execution_time()

        # Get missing data percentages
        missing_data = diagnostics.missing_data()

        # Get outdated packages
        outdated_packages = diagnostics.outdated_packages_list()

        return jsonify({
            "execution_time": {
                "ingestion_time": execution_times[0],
                "training_time": execution_times[1]
            },
            "missing_data": missing_data,
            "outdated_packages": outdated_packages
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#######################Root Endpoint
@app.route("/", methods=['GET'])
def home():
    """
    Root endpoint that provides information about the API and available endpoints.
    """
    return jsonify({
        "message": "Welcome to the Dynamic Risk Assessment System API",
        "endpoints": {
            "/": "This help message",
            "/prediction": "POST - Make predictions using the deployed model",
            "/scoring": "GET - Get the F1 score of the deployed model",
            "/summarystats": "GET - Get summary statistics for the ingested data",
            "/diagnostics": "GET - Get diagnostic information about the model and data"
        }
    })

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
