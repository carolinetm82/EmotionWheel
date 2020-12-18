from dash.dependencies import Input, Output, State
from layouts import df1,df2,urllib,dash_table,joblib,filename1
from app import app
from functions import px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Creating a callback so we can replace a dataframe table by another table with the dropdown option
@app.callback(
    Output('table-raw', 'children'),
    Input('demo-dropdown', 'value'))
def update_table(value):
    if value == 'df1':
        df= df1
    elif value == 'df2':
        df=df2
    table=dash_table.DataTable(
          css=[{'selector': '.row', 'rule': 'margin: 0'}],    
          data=df.to_dict('records'),
          columns=[{'id': c, 'name': c} for c in df.columns],
          fixed_rows={'headers': True},
          page_size=50,
          style_data={'whiteSpace': 'normal', 'height': 'auto'},
          style_header={'backgroundColor': 'rgb(193, 205, 205)'},
          style_cell={'maxWidth': '400px','backgroundColor': 'rgb(202, 225, 255)','color': 'rgb(0, 0, 139)','fontSize':12, 'font-family':'sans-serif'},
          style_table={'height': '300px','overflowY': 'auto'},
          id='table1'
          )
    return table

# Creating a callback calling an Emotion predictor
@app.callback(
    Output("example-output", "children"), [Input("example-button", "n_clicks")],[State("input", "value")]
)
def output_text(n,value):
    if (value is None) or (value==''):
        return "Please type a word or text to get the emotion"
    else:    
        emotion=joblib.load(filename1).predict([value])
        return emotion


