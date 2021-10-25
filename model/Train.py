import joblib
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import requests
import json


def train_model():
    iris_df = datasets.load_iris()

    x = iris_df.data
    y = iris_df.target

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    dt = LogisticRegression(random_state=0, solver='liblinear').fit(X_train, y_train)
    preds = dt.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    joblib.dump(dt, 'iris-model.model')
    print(f'Model Training Finished.\n\tAccuracy obtained: {accuracy}')


# heroku url
heroku_url = 'https://iris-prediction-flask.herokuapp.com/'  # change to your app name

# test data
data = {'sepal_length': 7.2
    , 'sepal_width': 5.3
    , 'petal_length': 2.5
    , 'petal_width': 1.1}

data = json.dumps(data)
print(data)

# check response code
r_survey = requests.post(heroku_url, data)
print(r_survey)

send_request = requests.post(heroku_url, data)
print(send_request)
print(send_request.json())
