import pandas as pd
import numpy as np
import nltk
from collections import defaultdict
from time import time
import joblib
from joblib import load


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff


df1 = pd.read_csv('data/Emotion_final.csv')
df2 = pd.read_csv('data/text_emotion.csv')


# Create figure 1 : histogram of emotions
fig1=px.histogram(df1,x='Emotion')
fig1.update_layout(title='Count of texts by emotion in the Kaggle Dataset')


# Create figure 2 : histogram of words
# Remind which the corpus and the targets are
corpus= np.array(df1['Text'])
targets = np.array(df1['Emotion'])

stopwords = set(nltk.corpus.stopwords.words("english"))

# Vobabulary analysis
vec = CountVectorizer(stop_words=stopwords)
X = vec.fit_transform(corpus)
words = vec.get_feature_names()


# Compute rank
wsum = np.array(X.sum(0))[0]
ix = wsum.argsort()[::-1]
wrank = wsum[ix] 
labels = [words[i] for i in ix]

# Sub-sample the data to plot.
# take the 20 first
def subsample(x):
    return np.hstack(x[:20])

freq = subsample(wrank)
r = np.arange(len(freq))

fig2 = go.Figure(go.Bar(x=r, y=freq))
fig2.update_traces(marker_color='lightsalmon')
fig2.update_xaxes(
        tickmode='array',
        tickvals = r,
        ticktext = labels
)

fig2.update_layout(title='The 20 most frequent words of the Kaggle Dataset',
                   xaxis_title="Words",
                   yaxis_title="Frequency")


# All the figures below will use the vect-logit

# Logistic Regression



X_train, X_test, y_train, y_test = train_test_split(corpus, targets, test_size=0.2,random_state=42)

filename1='finalized_regression.sav'
y_scores = joblib.load(filename1).predict_proba(X_test)

# One hot encode the labels in order to plot them
y_onehot = pd.get_dummies(y_test, columns=joblib.load(filename1).classes_)


# Create a multiclass ROC curve
# Create an empty figure, and iteratively add new lines
# every time we compute a new class
fig3 = go.Figure()
fig3.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)

for i in range(y_scores.shape[1]):
    y_true = y_onehot.iloc[:, i]
    y_score = y_scores[:, i]

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_score = average_precision_score(y_true, y_score)

    name = f"{y_onehot.columns[i]} (AP={auc_score:.2f})"
    fig3.add_trace(go.Scatter(x=recall, y=precision, name=name, mode='lines'))

fig3.update_layout(
    title ='Multiclass ROC curve',
    xaxis_title='Recall',
    yaxis_title='Precision',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=700, height=500
)

# Create Precision-Recall Curve
fig4 = go.Figure()
fig4.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

for i in range(y_scores.shape[1]):
    y_true = y_onehot.iloc[:, i]
    y_score = y_scores[:, i]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
    fig4.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

fig4.update_layout(
    title ='Precision-Recall curve',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=700, height=500
)

# Confusion matrix

y_pred = joblib.load(filename1).predict(X_test)
cmat = confusion_matrix(y_test, y_pred)
x=list(np.unique(targets))
y=list(np.unique(targets))

fig5 = ff.create_annotated_heatmap(cmat, x=x, y=y, colorscale='Viridis')
fig5.update_layout(title='Confusion matrix')
fig5.update_yaxes(autorange='reversed')