from flask import Flask, request, jsonify, render_template
import pickle

# Load the trained model using pickle
with open(f'model/svm', 'rb') as f:
    svm_model = pickle.load(f)

# Load the TfidfVectorizer used during training
with open(f'model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')

# Define a route to test the model with new data
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))

    if request.method == 'POST':
        email_text = request.form['emailText'] if 'emailText' in request.form else ""
        return render_template('main.html', email_text=email_text)
        # email_text = ''.join(email_text.splitlines())
        # print(email_text)
        # # Transform the email text using the loaded TfidfVectorizer
        # # Make the prediction using the loaded model
        # predictionTitle = "This email is:\n"
        # result = svm_model.predict(vectorizer.transform([email_text]))
        
        # # Return the prediction result as JSON
        # return render_template('main.html', email_text=email_text, predictionTitle=predictionTitle, result=result)
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True)