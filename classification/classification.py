import pandas as pd
import numpy as np
from sklearn import tree, preprocessing
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import pickle
from data_exploration import Exploration, Modeling
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, render_template, make_response
from flask_wtf import FlaskForm
from wtforms import StringField, validators
from wtforms.validators import DataRequired, Email
import io
from flask_restful import Resource, Api

DEBUG = True
app = Flask(__name__)
# app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'abcdefgh'
api = Api(app)

class TextFieldForm(FlaskForm):
    text = StringField('Document Content', validators=[validators.data_required()])


class Main(Resource):
    def __init__(self):
        print('Executing Main Funciton')
        self.exp = Exploration()
        self.model = Modeling()
        self.create_models()
        self.labels = ['APPLICATION', 'BILL', 'BILL BINDER', 'BINDER', 'CANCELLATION NOTICE', 'CHANGE ENDORSEMENT',
                       'DECLARATION', 'DELETION OF INTEREST', 'EXPIRATION NOTICE', 'INTENT TO CANCEL NOTICE',
                       'NON-RENEWAL NOTICE', 'POLICY CHANGE', 'REINSTATEMENT NOTICE', 'RETURNED CHECK']

    def create_models(self):

        # Remove NA in Dataset
        self.df = self.exp.remove_na()

        # Lable Encoding
        self.df, self.labels, self.doc = self.exp.label_encoder()

        # create train test validation sets
        self.model.create_test_validation_train(self.df)

        # Create Feature vectors using Term Frequency and Inverse document frequency
        self.model.create_vectors()

        # Training Classifiers

        # Logistic Regression
        self.clf = LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', C=14)
        self.model.train_classifier(self.clf)

        # Decision Tree
        # self.clf = tree.DecisionTreeClassifier()
        # self.model.train_classifier(self.clf)

        # SVM
        # n_estimators = 10
        # clf = OneVsRestClassifier(
        #     BaggingClassifier(SVC(kernel='linear', probability=True), max_samples=1.0 / n_estimators,
        #                       n_estimators=n_estimators))
        # self.model.train_classifier(self.clf)

        # K-Nearest Neighbors
        # knn = KNeighborsClassifier(n_neighbors=1)
        # self.model.train_classifier(self.clf)


class Flask_Work(Resource):
    def __init__(self):
        self.labels = ['APPLICATION', 'BILL', 'BILL BINDER', 'BINDER', 'CANCELLATION NOTICE', 'CHANGE ENDORSEMENT',
                       'DECLARATION', 'DELETION OF INTEREST', 'EXPIRATION NOTICE', 'INTENT TO CANCEL NOTICE',
                       'NON-RENEWAL NOTICE', 'POLICY CHANGE', 'REINSTATEMENT NOTICE', 'RETURNED CHECK']

    def get(self):
        """
        This method will render the index.html page
        :return: return to index.html
        """
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'), 200, headers)

    def post(self):
        """

        :return: Confidence and Label Type
        """
        f = open('machine_learning_lr.pkl', 'rb')
        logistic_regression_clf = pickle.load(f)
        tfidf = TfidfVectorizer()
        transformer = TfidfTransformer()
        l_map = {k: v for k, v in enumerate(self.labels)}
        words = request.form['Document Content']
        document = pd.DataFrame([words])
        document.columns = ["doc"]
        print(document['doc'])
        tfidf_test = tfidf.transform(document['doc'])
        tfidf_test_vector = transformer.fit_transform(tfidf_test)
        prediction = logistic_regression_clf.predict(tfidf_test_vector)
        confidence_interval = logistic_regression_clf.predict_proba(tfidf_test_vector)
        calculate_confidence = []
        total = confidence_interval[0].sum()
        for each in confidence_interval:
            for each1 in each:
                calculate_confidence.append(each1/total)
        print(l_map[prediction[0]])
        confidence = max(calculate_confidence)
        value = {
            'Confidence': confidence,
            'Label': 'The Type of the document is {}'.format(l_map[prediction[0]])
        }

        return {
            'Confidence': confidence,
            'Label': 'The Type of the document is {}'.format(l_map[prediction[0]])
        }

api.add_resource(Flask_Work, '/', '/penki')


if __name__ == '__main__':
    Main()
    app.run(host='127.0.0.1', port=4000, debug=True)
