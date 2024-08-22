from flask import Flask, request, render_template 
import pickle
import numpy as np

##############################
#   Load the Trained Models  #
##############################
# Load the trained SVM model using pickle
with open(f'model/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Load the trained Random Forest model using pickle
with open(f'model/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Load the trained Logistic Regression model using pickle
with open(f'model/logistic_model.pkl', 'rb') as f:
    log_reg_model = pickle.load(f)
    
# Load the trained Naive Bayes model using pickle
with open(f'model/naive_bayes_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

# Load the trained XGBoost model using pickle
with open(f'model/xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Load the TfidfVectorizer used during training
with open(f'model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess_text(text):
    # Vectorize the input text using the loaded vectorizer
    return vectorizer.transform([text])

# Define the label mapping dictionary
label_mapping = {0: "ham", 1: "spam"}

# Convert textual predictions to numerical values
reverse_label_mapping = {"ham": 0, "spam": 1}

def ensemble_predictions_with_confidence(vectorized_text):
    models = {
        "Naive Bayes": nb_model,
        "Logistic Regression": log_reg_model,
        "Random Forest": rf_model,
        "SVM": svm_model,
        "XGBoost": xgb_model
    }

    predictions = {}

    for model_name, model in models.items():
        pred = model.predict(vectorized_text)[0]

        # Convert textual predictions to numerical if necessary
        if pred in reverse_label_mapping:
            pred = reverse_label_mapping[pred]

        prob = model.predict_proba(vectorized_text)[0]

        confidence = max(prob) * 100  # Convert to percentage

        # Map the prediction to "ham" or "spam" using the label_mapping
        predictions[model_name] = {"Prediction": label_mapping[pred], "Confidence": confidence}

    # Display predictions and confidence levels
    for model_name, result in predictions.items():
        print(f"{model_name}: Prediction = {result['Prediction']}, Confidence = {result['Confidence']:.2f}%")

    # Count the number of 'spam' votes
    spam_count = sum(1 for result in predictions.values() if result['Prediction'] == 'spam')

    # Final decision based on majority vote
    if spam_count >= 3:
        final_prediction = "Spam"
    else:
        final_prediction = "Non-Spam"

    return final_prediction


########################
#   Application Setup  #
########################
# Initialize the Flask app
app = Flask(__name__, template_folder='templates')

# Define a route to test the model with new data
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))

    if request.method == 'POST':
        email_text = request.form.get('emailText') if 'emailText' in request.form else ""
        vectorized_text = preprocess_text(email_text)
        result = ensemble_predictions_with_confidence(vectorized_text)

        # Return the prediction result as JSON
        return render_template('main.html', 
                               result = result)
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True)