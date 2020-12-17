import pandas as pd
import numpy as np
import nltk
from collections import defaultdict
from time import time
import joblib
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB


df1 = pd.read_csv('data/Emotion_final.csv')

corpus= np.array(df1['Text'])
targets = np.array(df1['Emotion'])


# We create a pipeline with CountVectorizer and Logistic Regression

stopwords = set(nltk.corpus.stopwords.words("english"))

# Logistic Regression
pipe2 = Pipeline([
    ('vect', CountVectorizer(stop_words=stopwords)),
    ('logit', LogisticRegression(max_iter=1000)),
])


# Define the inputs and outputs
X = corpus
y = targets

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(corpus, targets, test_size=0.2,random_state=42)

# Fit the model
model = pipe2
model.fit(X_train, y_train)

# Save the model on filename1
filename1 = 'finalized_regression.sav'
joblib.dump(model, filename1)