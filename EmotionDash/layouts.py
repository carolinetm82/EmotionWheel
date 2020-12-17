import dash_core_components as dcc
import dash_html_components as html
import dash_table
import urllib
import joblib
from joblib import load
from functions import *


# Create an index page for homepage
index_page = html.Div([
    html.H1('Welcome to the dashboard of the Wheel of Emotions'),
    dcc.Markdown('''
    In this dashboard we will only see the results of the raw data from Kaggle,
    we only removed the stopwords, there is no other preprocessing.
    '''),    
])

# Create page 1
layout1 = html.Div([
    html.H4(children='The Wheel of Emotions'),
    # Create the data table containing the universities dataset
    dcc.Tabs([
        dcc.Tab(label='Kaggle', children=[
          dash_table.DataTable(
          css=[{'selector': '.row', 'rule': 'margin: 0'}],    
          data=df1.to_dict('records'),
          columns=[{'id': c, 'name': c} for c in df1.columns],
          fixed_rows={'headers': True},
          page_size=30,
          style_data={'whiteSpace': 'normal', 'height': 'auto'},
          style_header={'backgroundColor': 'rgb(193, 205, 205)'},
          style_cell={'maxWidth': '400px','backgroundColor': 'rgb(202, 225, 255)','color': 'rgb(0, 0, 139)','fontSize':12, 'font-family':'sans-serif'},
          style_table={'height': '300px','overflowY': 'auto'},
          id='table1'
          )]
        ),
    
        dcc.Tab(label='data.world', children=[
          dash_table.DataTable(
          css=[{'selector': '.row', 'rule': 'margin: 0'}],    
          data=df2.to_dict('records'),
          columns=[{'id': c, 'name': c} for c in df2.columns],
          fixed_rows={'headers': True},
          page_size=30,
          style_data={'whiteSpace': 'normal', 'height': 'auto'},
          style_header={'backgroundColor': 'rgb(193, 205, 205)'},
          style_cell={'maxWidth': '400px','backgroundColor': 'rgb(202, 225, 255)','color': 'rgb(0, 0, 139)','fontSize':12, 'font-family':'sans-serif'},
          style_table={'height': '300px','overflowY': 'auto'},
          id='table2'
          )] 
        )
    ]),
    html.Br(),
    html.Br(),
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
    dcc.Tab(label='res', children=[
          dash_table.DataTable(
          css=[{'selector': '.row', 'rule': 'margin: 0'}],    
          data=printTable.to_dict('records'),
          columns=[{'id': c, 'name': c} for c in printTable.columns],
          page_size=30,
          style_data={'whiteSpace': 'normal', 'height': 'auto'},
          style_header={'backgroundColor': 'rgb(193, 205, 205)'},
          style_cell={'maxWidth': '400px','backgroundColor': 'rgb(202, 225, 255)','color': 'rgb(0, 0, 139)','fontSize':12, 'font-family':'sans-serif'},
          style_table={'height': '300px'},
          id='table3'
          )] 
        ),
    dcc.Markdown('''
    For the figures below we will only use the best classifier we found: the pipeline (Countvectorizer and Logistic Regression)

    '''),  
    dcc.Graph(
        id='roc-curve',
        figure=fig3
    ),
    dcc.Graph(
        id='tpr-fpr-curve',
        figure=fig4
    ),
    dcc.Graph(
        id='matrix',
        figure=fig5
    ),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])

