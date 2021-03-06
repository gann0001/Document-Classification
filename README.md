﻿*HeavyWater Document Classification:*

**Problem Description:**

The problem consists of predicting the type of the document based on hashed
document data. This problem is a multi-class classification problem with
14 target variables and hashed document words.

**List of assumptions:**

The dataset words are the output of the OCR stage of the data pipeline. Hence
words contain stop words which are also hashed and words in the document
are in order

**Data Description:**

The data consists of 14 document labels and hashed document content.

**Exploratory Data Analysis:**

Below is the count of documents for each document label. I observed BILL
and POLICY Change contains a greater number of documents whereas INTENT
TO CANCEL NOTICE and APPLICATION has a smaller number of documents


As the document content is hashed hence I could not perform more
operations on real data like removing stop words, stemming, and
lemmatization.

**Missing Values in Data:**

There are 45 records where the document content is missing


>$df.isna().sum()
>
>label 0
>doc 45

**Analysis Plan:**

**Feature Engineering:**

I used TFIDF (Term Frequency-Inverse Document Frequency) to create
feature vectors where is used Sklearn TFIDF vectorizer and transformer
to fit all training, test and validation vectors in the same dimensions.

Encoded the document labels to make categorical values discrete using
LabelEncoder

**Split the Data for the train, validation, and test**
I split the training data into three parts with the ration of 65, 15 and
15, the first part is to train the model, the second part of the data is to
validate the model, and the third part of the data to test the model. If
there is some overfitting or underfitting validation and testing will
perform abnormal

**Explanation of Modeling Choice:**

I tried various models like Logistic Regression, Random Forest, Support Vector Machine (SVM), Decision Trees, and K-nearest neighbors so that I can compare model performance metrics for better accuracy. 

**Metric Evaluation for Machine Learning Models:**

  >Method  |              Validation Data Accuracy |  Test Data Accuracy
  --------------------- -------------------------- --------------------
  >Logistic Regression |  0.856              | 		 0.85       
  >Decision Tree       |  0.77		     |            0.77           
  >Random Forest       |  0.8318   		|		 0.83                  
  >K Nearest Neighbors |  0.67			|		 0.66                    
  >Random Forest                                    
Logistic Regression performed better across all the modes. Hyperparameter tuning definitely helps to make the model better for random forest and SVM.

**Web Page Development:**

-   I used Flask which is a microframework written in python.

-   Restful API for the web service and it returns JSON formatted output

-   Used HTML/CSS/JavaScript/Bootstrap to develop basic webpage where it
    contains text box and submit button to take an action

**Deployment:**

-   Deployed web application in Amazon Webservice

-   I generated a pickle file for trained logistic regression
    classifier(machine\_learning\_lr.pkl) which helped me to deploy in
    production. I could not upload pickle file in GIT as it is having
    more than 100 MB

-   Below is the link for accessing web application hosted in AWS

> <http://18.191.6.105/>

**Execution Instructions:**

**Type 1 Simple Program execution:**

> **Step1**: Download the pickle file from below location and store in
> classification folder. Please let me know if you don’t have access.
>
> https://drive.google.com/file/d/1Q4u2sk9KXsmTkVhOtF8I-iBSQs8tzxgz/view?usp=sharing
>
>**Step2:** Install requirements.txt file
>
>Pip3 install -r requirements.txt
>
> **Step3:** Run below command to execute the python file
>
> Python3 classification.py

**Type 2:** To access the web application, you may open below link to
    test the document type. This web link contains text box and submit
    button

> <http://18.191.6.105/>

**Type 3:** As I used REST API, you may execute below command predict document type and confidence

> \$curl http://18.191.6.105/ -d "Document Content= words”

**Challenges Faced:**

1.  As the document content is hashed I could not able to do the feature
    engineering with best features to train the model

2.  In order to deploy in AWS, I tried various web servers Zappa, Nginix
    but I had experienced some difficulties in resolving errors.

**Improvements:**

1.  Hyper-Parameter tuning for support vector machine and XG Boost model
    definitely improve the accuracy, but it needs GPU’s to run the
    program

2.  Removing stop words, Stemming and Lemmatization will also improve
    the accuracy if I know the actual data.

3.  TFIDF vector created big dimension vector where most of the columns
    are zero if I delete the most common columns, again I can make
    model smarter. But handling huge dimension vectors again we need
    GPU’s.



