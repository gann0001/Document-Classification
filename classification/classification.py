import pandas as pd
import numpy as np
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
from data_exploration import Exploration, Modeling
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, validators
from wtforms.validators import DataRequired, Email

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'abcdefgh'

class TextFieldForm(FlaskForm):
    text = StringField('Document Content', validators=[validators.data_required()])


class main():
    def __init__(self):
        DEBUG = True
        # self.app = Flask(__name__)
        # self.app.config.from_object(__name__)
        # self.app.config['SECRET_KEY'] = 'abcdefghijkl'
        # self.app.run(host='127.0.0.1', port=4000)
        print('Executing Main Funciton')
        self.exp = Exploration()
        self.model = Modeling()
        # self.create_models()
        self.flask_work()
        self.labels = ['APPLICATION', 'BILL', 'BILL BINDER', 'BINDER', 'CANCELLATION NOTICE', 'CHANGE ENDORSEMENT',
                       'DECLARATION', 'DELETION OF INTEREST', 'EXPIRATION NOTICE', 'INTENT TO CANCEL NOTICE',
                       'NON-RENEWAL NOTICE', 'POLICY CHANGE', 'REINSTATEMENT NOTICE', 'RETURNED CHECK']


    def create_models(self):

        #Remove NA in Dataset
        self.df = self.exp.remove_na()

        #Lable Encoding
        self.df, self.labels, self.doc =  self.exp.label_encoder()

        #create train test validation sets
        self.model.create_test_validation_train(self.df)

        #Create vectors using tfidf vectorizer
        self.model.create_vectors()

        #train classifiers
        self.clf = LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', C=14)
        self.model.train_classifier(self.clf)
#
    def flask_work(self):
        @app.route('/')
        def hello():
            form = TextFieldForm(request.form)
            if request.method == 'POST':
                print('helloooooo')
                words = request.form['Document Content']
                print(words)
            return render_template('hello.html', form=form)

        @app.route('/', methods=['GET', 'POST'])
        def predict():
            f = open('../machine_learning.pkl', 'rb')
            vectorizer, transformer, logistic_regression_clf = pickle.load(f), pickle.load(f), pickle.load(f)
            print('in vect')
            l_map = {k: v for k, v in enumerate(self.labels)}
            words = request.form['Document Content']
            print(words)
            document = pd.DataFrame([words])
            document.columns = ["doc"]
            # vectorizer = TfidfVectorizer()
            # transformer = TfidfTransformer()
            tfidf_test = vectorizer.transform(document['doc'])
            tfidf_test_vector = transformer.fit_transform(tfidf_test)
            # print(features.shape)
            prediction = logistic_regression_clf.predict(tfidf_test_vector)
            print(prediction)
            print(l_map[prediction[0]])
            return '<h1> Your document is of type {} '.format(l_map[prediction[0]])




if __name__ == '__main__':
    main()
    app.run(host='127.0.0.1', port=4000)