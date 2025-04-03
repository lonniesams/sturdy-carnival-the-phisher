# app.py
from flask import Flask, request, render_template
# Import necessary classes from transformers and torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import operator # To find max value in dictionary if needed (alternative to user's lambda)

# Initialize Flask app
app = Flask(__name__)

# --- Load Tokenizer and Model (Using your provided code) ---
# Load them globally when the app starts
model_name = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Set the model to evaluation mode (important for inference)
    model.eval()
    print(f"Tokenizer and Model '{model_name}' loaded successfully.")
    # You can optionally print the model's expected labels if available in config
    # print(f"Model config labels (if available): {model.config.id2label}")
except Exception as e:
    print(f"Error loading tokenizer or model '{model_name}': {e}")
    tokenizer = None
    model = None # Flag that loading failed

# --- Prediction Function (Your provided code) ---
def predict_email(email_text):
    if not tokenizer or not model:
        raise RuntimeError("Tokenizer or Model not loaded.") # Should not happen if initial check passes

    # Preprocess and tokenize
    inputs = tokenizer(
        email_text,
        return_tensors="pt", # PyTorch tensors
        truncation=True,     # Truncate long emails
        max_length=512       # Max sequence length for the model
    )

    # Get prediction - no need to track gradients for inference
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply softmax to logits to get probabilities
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get probabilities for each class (index matters!)
    probs = predictions[0].tolist() # Get the probabilities for the first (only) input

    # --- Create labels dictionary ---
    # IMPORTANT: This assumes the model's output logits correspond to these labels IN THIS ORDER.
    # Verify this order based on the model card or model.config.id2label if possible.
    labels = {
        "Legitimate Email": probs[0],
        "Phishing Link Detected": probs[1], # Assuming 'phishing_url' means a bad link found
        "Legitimate Link Detected": probs[2], # Assuming 'legitimate_url' means a good link found
        "Phishing Link Detected (Alt)": probs[3] # Assuming 'phishing_url_alt' is also bad
    }

    # Determine the most likely classification based on highest probability
    # Using operator.itemgetter is slightly more standard than lambda for this case
    max_label_item = max(labels.items(), key=operator.itemgetter(1))

    return {
        "prediction": max_label_item[0],  # The label name with the highest probability
        "confidence": max_label_item[1],  # The highest probability value
        "all_probabilities": labels       # Dictionary of all labels and their probabilities
    }

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    email_text_input = ""
    error_message = None
    friendly_label_display = None # For simple display (e.g., Phishing/Legitimate)
    result_details = None # To hold the full dictionary from predict_email

    # Check if model loaded correctly at startup
    if not tokenizer or not model:
         error_message = "Phishing detection model could not be loaded. Please check the server logs."
         # Pass the error immediately to the template
         return render_template('index.html', error=error_message)

    if request.method == 'POST':
        email_text_input = request.form['text']
        if email_text_input:
            try:
                # Perform classification using your function
                result_details = predict_email(email_text_input)
                prediction_result = result_details['prediction'] # Get the top prediction label
                print(f"Input: '{email_text_input[:100]}...', Result: {result_details}") # Log detailed result

                # --- Determine a simple display label ---
                # Customize this logic based on how you want to interpret the model's specific labels
                if "Phishing Link" in prediction_result:
                     friendly_label_display = "Suspicious / Phishing Link Likely"
                elif "Legitimate" in prediction_result:
                     friendly_label_display = "Likely Legitimate"
                else:
                     friendly_label_display = prediction_result # Fallback to the raw label name

            except Exception as e:
                 print(f"Error during prediction: {e}")
                 error_message = f"An error occurred during analysis: {e}"

    # Render the HTML template
    return render_template(
        'index.html',
        result=result_details, # Pass the whole result dictionary
        friendly_label=friendly_label_display, # Pass the simplified label
        text=email_text_input,
        error=error_message
    )

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True) # Set debug=False for production