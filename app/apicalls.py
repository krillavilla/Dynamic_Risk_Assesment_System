import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Load config.json
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])

# Prepare test data file path for prediction
test_data_file = os.path.join(os.getcwd(), test_data_path, 'testdata.csv')

#Call each API endpoint and store the responses
# Call prediction endpoint
headers = {'Content-Type': 'application/json'}
data = {'file_path': test_data_file}
response1 = requests.post(f'{URL}/prediction', json=data, headers=headers).json()

# Call scoring endpoint
response2 = requests.get(f'{URL}/scoring').json()

# Call summary statistics endpoint
response3 = requests.get(f'{URL}/summarystats').json()

# Call diagnostics endpoint
response4 = requests.get(f'{URL}/diagnostics').json()

#combine all API responses
responses = {
    'prediction': response1,
    'scoring': response2,
    'summary_statistics': response3,
    'diagnostics': response4
}

#write the responses to your workspace
# Create model directory if it doesn't exist
os.makedirs(os.path.join(os.getcwd(), output_model_path), exist_ok=True)

# Write responses to apireturns.txt
with open(os.path.join(os.getcwd(), output_model_path, 'apireturns.txt'), 'w') as f:
    f.write(json.dumps(responses, indent=4))
