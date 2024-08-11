import flask
import pickle

# Use pickle to load in the pre-trained logistic regression model.
with open(f'model/logistic_regression.pkl', 'rb') as f:
    logRegModel = pickle.load(f)

# Use pickle to load in the pre-trained naive bayes model.
with open(f'model/naive_bayes.pkl', 'rb') as f:
    nbModel = pickle.load(f)

# Use pickle to load in the pre-trained random forest model.
with open(f'model/random_forest.pkl', 'rb') as f:
    rfModel = pickle.load(f)

# Use pickle to load in the pre-trained svm model.
with open(f'model/svm.pkl', 'rb') as f:
    svmModel = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        emailText = flask.request.form.get("emailText")
        prediction = logRegModel.predict(emailText)
        return(flask.render_template('main.html', emailCategory=prediction))

if __name__ == '__main__':
    app.run()