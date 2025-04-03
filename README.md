# Phishing Email Detector

## Description

This project is a web application built with Flask that analyzes email text content to detect potential phishing attempts. It utilizes a pre-trained machine learning model from the Hugging Face Transformers library to classify emails. [cite: 1]

## Features

* Provides a simple web interface to paste and analyze email text.
* Uses the `cybersectony/phishing-email-detection-distilbert_v2.4.1` model for classification. [cite: 1]
* Displays the most likely classification (e.g., "Likely Legitimate", "Suspicious / Phishing Link Likely"). [cite: 1]
* Shows the confidence score for the top prediction. [cite: 1]
* Provides detailed probabilities for all classification categories considered by the model. [cite: 1]

## Dependencies

* Python 3.x
* Flask [cite: 1]
* Hugging Face Transformers (`transformers`) [cite: 1]
* PyTorch (`torch`) [cite: 1]

You can install the Python dependencies using pip:
```bash
pip install Flask transformers torch
