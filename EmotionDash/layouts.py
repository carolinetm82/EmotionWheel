import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import urllib
import joblib
from joblib import load
from functions import *


# Create an index page for homepage that contains an emotion predictor
index_page = html.Div([
    html.H1('Welcome to the dashboard of the Wheel of Emotions'),
    dcc.Markdown('''
    In this dashboard we will only see the results of the raw data from Kaggle,
    we only removed the stopwords, there is no other preprocessing.
    '''), 
    html.Br(),
    html.Br(),
    dcc.Markdown('''
    **Emotions Predictor**
    '''),
    dbc.Input(id="input", placeholder="Type something...", type="text"),
    dbc.Button("Submit", id="example-button", className="mr-2"),
    html.Br(),
    html.P(id="output"),
    html.Span(id="example-output", style={"vertical-align": "middle"}),

])

# Create page 1
layout1 = html.Div([
    html.H4(children='Emotions Data Analysis'),
    # Create the data tables Kaggle and dataworld dataset 
    dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'Kaggle', 'value': 'df1'},
            {'label': 'dataworld', 'value': 'df2'},
        ],
        value='df1'
    ),
    html.Div(id='table-raw'), 
    html.Br(),
    html.Br(),
    # Adding histogram of Emotions and histogram of words
    dcc.Graph(
        id='count-emotions',
        figure=fig1
    ),
    dcc.Graph(
        id='count-words',
        figure=fig2
    ),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])

def print_table(res):
    # Compute the results
    final = {}
    for model in res:
        arr = np.array(res[model])
        final[model] = {
            "time" : arr[:, 0][0].round(2),
            "precision_score": arr[:,1][0].round(2),
            "recall_score":arr[:, 2][0].round(2),
            "f1_score":arr[:, 3][0].round(2),
        }

    df = pd.DataFrame.from_dict(final, orient="index").reset_index()
    return df

filename='finalized_model.sav'
with open(filename,'rb') as f:
    printTable=print_table(joblib.load(f))


# Create page 2
layout2 = html.Div([
    html.H3('Classifiers Results'),
    # Display classifier results
    dash_table.DataTable(
    css=[{'selector': '.row', 'rule': 'margin: 0'}],    
    data=printTable.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in printTable.columns],
    page_size=30,
    style_data={'whiteSpace': 'normal', 'height': 'auto'},
    style_header={'backgroundColor': 'rgb(193, 205, 205)'},
    style_cell={'maxWidth': '400px','backgroundColor': 'rgb(202, 225, 255)','color': 'rgb(0, 0, 139)','fontSize':12, 'font-family':'sans-serif'},
    style_table={'height': '200px'},
    id='table3'
    ),
    dcc.Markdown('''
    For the figures below we will only use the best classifier we found: the pipeline (Countvectorizer and Logistic Regression)

    '''),  
    dcc.Graph(
        id='roc-curve',
        figure=fig3
    ),
    html.Div([
    dbc.Row(
        [
            dbc.Col(html.Div(dcc.Graph(id='tpr-fpr-curve',figure=fig4))),
            dbc.Col(html.Div(dcc.Markdown('''The area under curve(AUC) summarizes the skill of a 
            model across thresholds whereas the F1-score summarizes model skill for a specific probability threshold.  
            For each emotion we have an AUC of 0.99 
            or 0.98 which means their predictions are 98% or 99% correct 
            and the model has an excellent skill''')), width=4),
        ]
    ),
    ]),
    dcc.Graph(
        id='matrix',
        figure=fig5
    ),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])

