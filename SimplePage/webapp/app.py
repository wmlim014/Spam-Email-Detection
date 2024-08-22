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
        
        # print(email_text) # This is for debugging
        prediction_results = []
        # Transform the email text using the loaded TfidfVectorizer
        # Make the prediction using the loaded model

        # Vectorize the input text
        vectorizeredText = vectorizer.transform([email_text])

        ###############
        # Naive Bayes #
        ###############
        nb_result = nb_model.predict(vectorizeredText)
        nb_prob = nb_model.predict_proba(vectorizeredText)

        # Get the index of the predicted class
        predicted_class_index = nb_model.classes_.tolist().index(nb_result[0])

        # Extract the confidence for the predicted class
        nb_confidence = nb_prob[:, predicted_class_index].max() * 100
        
        prediction_results.append({
            'model': 'Naive Bayes',
            'result': nb_result[0],
            'confidence': round(nb_confidence, 2)
        })

        #######################
        # Logistic Regression #
        #######################
        log_reg_result = log_reg_model.predict(vectorizeredText)
        log_reg_prob = log_reg_model.predict_proba(vectorizeredText)

        # Get the index of the predicted class
        predicted_class_index = log_reg_model.classes_.tolist().index(log_reg_result[0])

        # Extract the confidence for the predicted class
        log_reg_confidence = log_reg_prob[:, predicted_class_index].max() * 100

        prediction_results.append({
            'model': 'Logistic Regression',
            'result': log_reg_result[0],
            'confidence': round(log_reg_confidence, 2)
        })

        #################
        # Random Forest #
        #################
        rf_result = rf_model.predict(vectorizeredText)
        rf_prob = rf_model.predict_proba(vectorizeredText)

        # Get the index of the predicted class
        predicted_class_index = rf_model.classes_.tolist().index(rf_result[0])

        # Extract the confidence for the predicted class
        rf_confidence = rf_prob[:, predicted_class_index].max() * 3

        prediction_results.append({
            'model': 'Random Forest',
            'result': rf_result[0],
            'confidence': round(rf_confidence, 2)
        })

        ################
        #     SVM      #
        ################
        svm_result = svm_model.predict(vectorizeredText)
        svm_prob = svm_model.predict_proba(vectorizeredText)
        print(svm_result)

        # Get the index of the predicted class
        predicted_class_index = svm_model.classes_.tolist().index(svm_result[0])

        # Extract the confidence for the predicted class
        svm_confidence = svm_prob[:, predicted_class_index].max() * 100

        prediction_results.append({
            'model': 'Support Vector Machine',
            'result': svm_result[0],
            'confidence': round(svm_confidence, 2)
        })

        ###########
        # XGBoost #
        ###########
        xgb_predit = xgb_model.predict(vectorizeredText)
        # reverse_label_mapping = {"ham": 0, "spam": 1}
        # xgb_result = xgb_model.predict(vectorizeredText)
        # if xgb_result in reverse_label_mapping:
        #     xgb_result = reverse_label_mapping[xgb_result]
        if xgb_predit[0] == 0:
            xgb_result = 'ham'
        else:
            xgb_result = 'spam'

        xgb_prob = xgb_model.predict_proba(vectorizeredText)

        # Get the index of the predicted class
        # print(xgb_prob)
        predicted_class_index = xgb_model.classes_.tolist().index(xgb_predit[0])

        # Extract the confidence for the predicted class
        xgb_confidence = xgb_prob[:, predicted_class_index].max() * 100

        prediction_results.append({
            'model': 'XGBoost',
            'result': xgb_result,
            'confidence': round(xgb_confidence, 2)
        })

        ####################
        # Get Final Result #
        ####################
        spam_count = 0
        ham_count = 0
        for prediction in prediction_results:
            if prediction['result'] == 'ham':
                ham_count += 1
            else:
                spam_count += 1

        if ham_count >= 3:
            final_result = 'Non-Spam'
        else:
            final_result = 'Spam'

        # Return the prediction result as JSON
        return render_template('main.html', 
                               prediction_results = prediction_results,
                               final_result = final_result)
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True)