
# 🧠 Dynamic Risk Assessment System

Hi, I'm Kashad J. Turner-Warren, and this repository contains my project for building a **Dynamic Risk Assessment System**. The goal of this system is to create an automated ML pipeline that evaluates corporate data and updates predictions based on new information, model drift, and data integrity—all while supporting modular deployment through API endpoints.

This project aligns with my focus on integrating machine learning with DevOps practices, and brings together skills in model training, deployment, diagnostics, and API development.

---

## 🚀 Project Overview

The system is built as a modular pipeline, covering the full ML lifecycle:

- **Data Ingestion**: Automatically ingests and merges CSV files from the `/sourcedata/` folder, deduplicates the dataset, and stores both the final data and a log of ingested files.
- **Model Training**: Trains a classification model on the dataset and saves it using Pickle.
- **Scoring**: Evaluates the model using the F1 score metric and stores the result.
- **Deployment**: Moves the trained model, scoring results, and ingestion records into a production-ready folder.
- **Diagnostics**: Measures performance latency, summary statistics, missing data percentage, and environment dependency mismatches.
- **Reporting**: Generates a confusion matrix to visualize model accuracy on test data.
- **API**: Provides endpoints for predictions, scoring, diagnostics, and statistics using Flask.
- **Process Automation**: A master script checks for new data or model drift and triggers retraining and redeployment if needed.

---

## 🧰 Tech Stack

- Python 3
- scikit-learn
- pandas
- Flask
- NumPy
- Pickle
- OS & JSON for configuration
- pytest (for testing)
- cron (for automation)

---

## 📁 Project Structure

```plaintext
.
├── apicalls.py
├── app.py
├── config.json
├── deployment.py
├── diagnostics.py
├── fullprocess.py
├── ingestion.py
├── models/
├── output/
├── practicemodels/
├── production_deployment/
├── reporting.py
├── requirements.txt
├── scoring.py
├── sourcedata/
├── testdata/
├── training.py
├── wsgi.py
```

---

## 📊 API Endpoints

- `/prediction` – Returns predictions for a given dataset
- `/scoring` – Returns the latest model F1 score
- `/summarystats` – Returns summary statistics (mean, median, mode)
- `/diagnostics` – Returns diagnostics including timing, missing data, and dependency status

---

## ✅ Features

- Handles dynamic data updates automatically
- Monitors model performance for drift
- Provides API access for external services
- Includes full testing and diagnostics
- Designed for production deployment

---

## 💡 Why I Built This

I built this as part of my Udacity Machine Learning DevOps Nanodegree program. My goal was to strengthen my understanding of ML lifecycle management, pipeline automation, and scalable deployment—skills I plan to apply in offensive security and AI-driven cybersecurity research.

---

## 📌 Next Steps

- Add Docker support for containerization
- Integrate with a cloud service like AWS or GCP for remote model serving
- Implement logging and exception tracking with a tool like Sentry or Prometheus

---

## 🙌 Acknowledgements

Special thanks to the Udacity community and instructors for guidance and support. Also, shoutout to the amazing mentors and peers who gave feedback and helped me grow through this journey.

---

## 📬 Contact

Want to collaborate or chat? Feel free to reach out:

**Kashad J. Turner-Warren**  
[LinkedIn](https://www.linkedin.com/in/kashad-turner-warren) | [Twitter](https://twitter.com/KashadWarren) | [GitHub](https://github.com/kashadtw)

