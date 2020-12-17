import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app,dbc
from app import server
from layouts import layout1, layout2, index_page
import callbacks


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#ffe1ff",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "16rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Analysis of Emotions", className="display-4"),
        html.Hr(),
        html.P(
            "Navigation links", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", id="page-0-link"),
                dbc.NavLink("Page 1", href="/apps/app1", id="page-1-link"),
                dbc.NavLink("Page 2", href="/apps/app2", id="page-2-link"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    content
])

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(0, 3)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 0 as the homepage / index
        return True, False,False
    return [pathname == f"/page-{i}" for i in range(0, 3)]

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/apps/app1':
         return layout1
    elif pathname == '/apps/app2':
         return layout2
    else:
         return index_page

if __name__ == '__main__':
    app.run_server(debug=True)