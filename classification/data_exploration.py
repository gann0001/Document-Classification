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
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score


class Exploration:
    def __init__(self):
        print('Exploration Started')
        # Read Dataset using Pandas
        self.data = pd.read_csv('../shuffled-full-set-hashed.csv')

        # Create dataframe with column names
        self.df = pd.DataFrame(self.data.values, columns=['label', 'doc'])

    def remove_na(self):
        #Remove NA's from the dataframe
        self.df = self.df.dropna()
        return self.df

    def label_encoder(self):
        #Label Encoding to given labels
        self.labels = self.df.iloc[:, 0].values
        self.docs = self.df.iloc[:, 1].values
        le = preprocessing.LabelEncoder()
        self.df['label'] = le.fit_transform(self.df['label'])
        return self.df, self.labels, self.docs


class Modeling:
    def __init__(self):
        print('Models development started')

    def create_test_validation_train(self, df):
        """
        Dividing Train, Validation and Test to analyze my machine learning model on basis of
         over fitting or under fitting. Hence my accuracies or confusion matrix should be good
         for both vaidation and testing
        divided the samples to
        :param df: DataFrame consists of labels and docs
        :return: returning train, validation and test series of given samples
        """
        test_size = 0.15
        valid_size = 0.15

        self.X_train_test, self.X_valid, self.y_train_test, self.y_valid = train_test_split(df['doc'], df['label'], test_size=valid_size)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train_test, self.y_train_test, test_size=test_size)

        return self.X_train, self.X_test, self.X_valid, self.y_valid, self.y_train, self.y_test

    def create_vectors(self):
        """
        creating features from given samples using Term Frequency and Inverse document frequency(TFIDF)
        :return: returning feature vectors
        """
        print('Vectors')
        vectorizer = TfidfVectorizer()
        tfidf_train = vectorizer.fit_transform(self.X_train)
        tfidf_valid = vectorizer.transform(self.X_valid)
        tfidf_test = vectorizer.transform(self.X_test)


        transformer = TfidfTransformer()
        self.tfidf_train_vectors = transformer.fit_transform(tfidf_train)
        self.tfidf_valid_vectors = transformer.fit_transform(tfidf_valid)
        self.tfidf_test_vectors = transformer.fit_transform(tfidf_test)

        return self.tfidf_train_vectors, self.tfidf_valid_vectors, self.tfidf_test_vectors

    def train_classifier(self, clf):
        """
        training classifier features with target labels
        :param clf: machine learning model
        """
        print('train classifier')
        self.clf_fit = clf.fit(self.tfidf_train_vectors, self.y_train)
        self.predict_labels()

    def predict_labels(self):
        """
        predicting labels for Validation and Testing
        """
        print('predict labels')
        self.clf_pred_val = self.clf_fit.predict(self.tfidf_valid_vectors)
        self.clf_pred_test = self.clf_fit.predict(self.tfidf_test_vectors)
        self.validate_metrics()

    def validate_metrics(self):
        """

        :metrics: Confustion matrix, accuracy, accuracy with normalize
        """
        print('validate metrics')
        #Validation Set
        print('Confusion Matrix:\n',confusion_matrix(self.y_valid, self.clf_pred_val))
        print('Accuracy:',accuracy_score(self.y_valid, self.clf_pred_val))
        print('Accuracy with normalization:',accuracy_score(self.y_valid, self.clf_pred_val, normalize=False))
        print('F1 Score', f1_score(self.y_valid, self.clf_pred_val, average='micro'))

        #Test Set
        print('Confusion Matrix:\n', confusion_matrix(self.y_test, self.clf_pred_test))
        print('Accuracy:', accuracy_score(self.y_test, self.clf_pred_test))
        print('Accuracy with normalization:', accuracy_score(self.y_test, self.clf_pred_test, normalize=False))
        print('F1 Score:', f1_score(self.y_test, self.clf_pred_test, average='micro'))