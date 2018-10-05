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
from data_exploration import Exploration, Modeling
from sklearn.linear_model import LogisticRegression

class main:
    def __init__(self):
        print('Executing Main Funciton')
        self.exp = Exploration()
        self.model = Modeling()
        self.create_models()


    def create_models(self):

        #Remove NA in Dataset
        self.df = self.exp.remove_na()

        #Lable Encoding
        self.df, self.labels, self.doc =  self.exp.label_encoder()

        #create train test validation sets
        self.X_train, self.X_test, self.X_valid, self.y_valid, self.y_train, self.y_test = self.model.create_test_validation_train(self.df)

        #Create vectors using tfidf vectorizer
        self.tfidf_train_vectors, self.tfidf_valid_vectors, self.tfidf_test_vectors = self.model.create_vectors()

        #train classifiers
        self.clf = LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', C=14)
        self.model.train_classifier(self.clf)



if __name__=="__main__":
    main()
