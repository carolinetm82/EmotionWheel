import pandas as pd
import numpy as np
import nltk
from collections import defaultdict
from time import time
import joblib
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB


df1 = pd.read_csv('data/Emotion_final.csv')
df2 = pd.read_csv('data/text_emotion.csv')

# We create a function capable of training and testing our dataset with each classifier

stopwords = set(nltk.corpus.stopwords.words("english"))

def run_pipes(pipes, splits=10, test_size=0.2, seed=42):  
    res = defaultdict(list)
    spliter = ShuffleSplit(n_splits=splits, test_size=test_size, random_state=seed)
    for idx_train, idx_test in spliter.split(corpus):
        for pipe in pipes:
            # name of the model
            name = "-".join([x[0] for x in pipe.steps])
            
            # extract datasets
            X_train = corpus[idx_train]
            X_test = corpus[idx_test]
            y_train = targets[idx_train]
            y_test = targets[idx_test]
            
            # Learn
            start = time()
            pipe.fit(X_train, y_train)
            fit_time = time() - start
            
            # predict and save results
            y = pipe.predict(X_test)
            #print(name)
            #print(classification_report(y_test, y))
            res[name].append([
                fit_time,
                precision_score(y_test, y,average='weighted'),
                recall_score(y_test, y,average='weighted'),
                f1_score(y_test, y,average='weighted'),      
            ])
            
    return res




corpus= np.array(df1['Text'])
targets = np.array(df1['Emotion'])

# Linear Support Vector Machines
pipe1 = Pipeline([
    ('vect', CountVectorizer(stop_words=stopwords)),
    ('svml', LinearSVC()),
])

# Logistic Regression
pipe2 = Pipeline([
    ('vect', CountVectorizer(stop_words=stopwords)),
    ('logit', LogisticRegression(max_iter=1000)),
])

# Multinomial Naive Bayes
pipe3 = Pipeline([
    ('vect', CountVectorizer(stop_words=stopwords)),
    ('mult_nb', MultinomialNB()),
])

# Complement Naive Bayes classifier 
pipe4 = Pipeline([
    ('vect', CountVectorizer(stop_words=stopwords)),
    ('compl_nb', ComplementNB()),
])

# Naive Bayes classifier for multivariate Bernoulli models
pipe5 = Pipeline([
    ('vect', CountVectorizer(stop_words=stopwords)),
    ('bern_nb', BernoulliNB()),
])

# run base pipes

res = run_pipes([pipe1, pipe2,pipe3,pipe4,pipe5], splits=1)
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(res, filename)


#df_res1=print_table(res1).reset_index()