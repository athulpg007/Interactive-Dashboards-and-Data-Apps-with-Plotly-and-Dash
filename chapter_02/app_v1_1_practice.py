import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
	dcc.Dropdown(id='color_dropdown', options=[{'label': color, 'value': color} for color in ['red', 'blue', 'green', 'yellow']]),
	html.Br(),
	html.Div(id='color_output')
])

@app.callback(Output('color_output', 'children'),
			  Input('color_dropdown', 'value'))
def display_selected_color(color):
	if color is None:
		color = 'nothing'
	return f"You selected {color}"


if __name__ == '__main__':
	app.run_server(debug=True)