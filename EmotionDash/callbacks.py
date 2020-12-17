from dash.dependencies import Input, Output
from layouts import df1,df2,urllib
from app import app
from functions import px



'''
@app.callback(
    Output('research-vs-world_rank', 'figure'),
    Input('dropdown-univ', 'value'))
def update_figure(selected_country):
    filtered_df = df2[df2.country == selected_country]

# Create the scatter plot
    fig1 = px.scatter(filtered_df, x="world_rank", y="research",
                 hover_name="university_name",
                 title="Research vs Rank for the top 50 universities (select a country with the scroll bar above)")

    fig1.update_layout(transition_duration=500)

    return fig1
'''