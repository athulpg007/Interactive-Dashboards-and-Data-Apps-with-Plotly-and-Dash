import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px

import numpy as np
import pandas as pd

planets = ['Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
launchers = ['Falcon Heavy Reusable', 'Falcon Heavy Expendable']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
	dbc.Col([
		html.Br(),
		html.H1('JPL-Purdue Planetary Mission Design Suite')
	], style={'textAlign': 'center'}),
	html.Br(),

	html.Div([
	dbc.Row([
		dbc.Col([
			dbc.NavbarSimple([
				dbc.DropdownMenu([
					dbc.DropdownMenuItem(planet, href=planet) for planet in planets
				], label='Search trajectories'),
			dbc.DropdownMenu([
					dbc.DropdownMenuItem(launcher, href=launcher) for launcher in launchers
				], label='Launch Performance', style={"marginLeft": "15px"}),
			], brand='Home', brand_href='/')
		], lg=4),



	])

	]),



])

if __name__ == '__main__':
	app.run_server(debug=True)
