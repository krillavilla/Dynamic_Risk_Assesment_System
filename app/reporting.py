import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import diagnostics
import io


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path'])


##############Function for generating PDF report
def generate_pdf_report():
    """
    Generate a PDF report containing the confusion matrix, summary statistics,
    model's F1 score, and list of ingested files.
    """
    # Create model directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), output_model_path), exist_ok=True)

    # Define the PDF path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(os.getcwd(), output_model_path, f'model_report_{timestamp}.pdf')

    # Create a PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Add title
    title = Paragraph("Model Performance Report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 20))

    # Add timestamp
    timestamp_text = Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
    elements.append(timestamp_text)
    elements.append(Spacer(1, 20))

    # Add confusion matrix
    elements.append(Paragraph("Confusion Matrix", styles['Heading2']))
    elements.append(Spacer(1, 10))

    # Get confusion matrix from score_model function
    # Pass from_pdf_report=True to avoid recursive calls
    cm = score_model(from_pdf_report=True)

    # Convert confusion matrix to a table
    cm_data = [['', 'Predicted Negative', 'Predicted Positive'],
               ['Actual Negative', cm[0][0], cm[0][1]],
               ['Actual Positive', cm[1][0], cm[1][1]]]

    cm_table = Table(cm_data)
    cm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.white),
        ('BACKGROUND', (1, 0), (2, 0), colors.lightgrey),
        ('BACKGROUND', (0, 1), (0, 2), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold')
    ]))

    elements.append(cm_table)
    elements.append(Spacer(1, 20))

    # Add confusion matrix image
    confusion_matrix_path = os.path.join(os.getcwd(), output_model_path, 'confusionmatrix.png')
    if os.path.exists(confusion_matrix_path):
        elements.append(Paragraph("Confusion Matrix Visualization", styles['Heading3']))
        elements.append(Spacer(1, 10))
        elements.append(Image(confusion_matrix_path, width=400, height=300))
        elements.append(Spacer(1, 20))

    # Add F1 score
    elements.append(Paragraph("Model Performance Metrics", styles['Heading2']))
    elements.append(Spacer(1, 10))

    # Get F1 score from latestscore.txt
    try:
        with open(os.path.join(os.getcwd(), prod_deployment_path, 'latestscore.txt'), 'r') as f:
            f1_score = float(f.read().strip())

        metrics_data = [['Metric', 'Value'],
                       ['F1 Score', f"{f1_score:.4f}"]]

        metrics_table = Table(metrics_data)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
        ]))

        elements.append(metrics_table)
        elements.append(Spacer(1, 20))
    except Exception as e:
        elements.append(Paragraph(f"Error retrieving F1 score: {str(e)}", styles['Normal']))
        elements.append(Spacer(1, 20))

    # Add summary statistics
    elements.append(Paragraph("Data Summary Statistics", styles['Heading2']))
    elements.append(Spacer(1, 10))

    try:
        # Get summary statistics from diagnostics
        summary_stats = diagnostics.dataframe_summary(use_db=False)

        # Create a table for each column's statistics
        for stat in summary_stats:
            col_name = stat['column']
            elements.append(Paragraph(f"Column: {col_name}", styles['Heading3']))
            elements.append(Spacer(1, 5))

            stats_data = [['Statistic', 'Value'],
                         ['Mean', f"{stat['mean']:.4f}"],
                         ['Median', f"{stat['median']:.4f}"],
                         ['Standard Deviation', f"{stat['std']:.4f}"]]

            stats_table = Table(stats_data)
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
            ]))

            elements.append(stats_table)
            elements.append(Spacer(1, 10))
    except Exception as e:
        elements.append(Paragraph(f"Error retrieving summary statistics: {str(e)}", styles['Normal']))
        elements.append(Spacer(1, 20))

    # Add missing data information
    elements.append(Paragraph("Missing Data Analysis", styles['Heading2']))
    elements.append(Spacer(1, 10))

    try:
        # Get missing data percentages from diagnostics
        missing_data = diagnostics.missing_data(use_db=False)

        # Create a table for missing data
        missing_data_rows = [['Column', 'Missing Data (%)']]
        for item in missing_data:
            missing_data_rows.append([item['column'], f"{item['percentage']:.2f}%"])

        missing_data_table = Table(missing_data_rows)
        missing_data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
        ]))

        elements.append(missing_data_table)
        elements.append(Spacer(1, 20))
    except Exception as e:
        elements.append(Paragraph(f"Error retrieving missing data information: {str(e)}", styles['Normal']))
        elements.append(Spacer(1, 20))

    # Add execution time information
    elements.append(Paragraph("Execution Time Analysis", styles['Heading2']))
    elements.append(Spacer(1, 10))

    try:
        # Get execution times from diagnostics
        execution_times = diagnostics.execution_time(use_db=False)

        # Create a table for execution times
        execution_time_rows = [['Process', 'Execution Time (seconds)']]
        execution_time_rows.append(['Data Ingestion', f"{execution_times[0]:.4f}"])
        execution_time_rows.append(['Model Training', f"{execution_times[1]:.4f}"])

        execution_time_table = Table(execution_time_rows)
        execution_time_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
        ]))

        elements.append(execution_time_table)
        elements.append(Spacer(1, 20))
    except Exception as e:
        elements.append(Paragraph(f"Error retrieving execution time information: {str(e)}", styles['Normal']))
        elements.append(Spacer(1, 20))

    # Add ingested files information
    elements.append(Paragraph("Ingested Files", styles['Heading2']))
    elements.append(Spacer(1, 10))

    try:
        # Get list of ingested files from ingestedfiles.txt
        with open(os.path.join(os.getcwd(), prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
            ingested_files = f.read().strip().split('\n')

        # Create a table for ingested files
        ingested_files_rows = [['File Name']]
        for file in ingested_files:
            ingested_files_rows.append([file])

        ingested_files_table = Table(ingested_files_rows)
        ingested_files_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
        ]))

        elements.append(ingested_files_table)
        elements.append(Spacer(1, 20))
    except Exception as e:
        elements.append(Paragraph(f"Error retrieving ingested files information: {str(e)}", styles['Normal']))
        elements.append(Spacer(1, 20))

    # Build and save the PDF
    doc.build(elements)

    print(f"PDF report generated: {pdf_path}")
    return pdf_path


##############Function for reporting
def score_model(from_pdf_report=False):
    """
    Calculate a confusion matrix using the test data and the deployed model
    and write the confusion matrix to the workspace.

    Args:
        from_pdf_report (bool): Flag to indicate if this function is being called from generate_pdf_report
                               to prevent recursive calls.

    Returns:
        numpy.ndarray: The confusion matrix
    """
    # Create model directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), output_model_path), exist_ok=True)

    # Load the deployed model
    model_file_path = os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl')
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

    # Calculate confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    # Save confusion matrix plot
    confusion_matrix_path = os.path.join(os.getcwd(), output_model_path, 'confusionmatrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Generate PDF report only if not called from generate_pdf_report
    if not from_pdf_report:
        try:
            generate_pdf_report()
        except Exception as e:
            print(f"Error generating PDF report: {str(e)}")

    return cm


if __name__ == '__main__':
    score_model()
