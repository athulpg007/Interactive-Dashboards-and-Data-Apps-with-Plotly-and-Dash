import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import numpy as np
import pandas as pd
from urllib.parse import unquote
from scipy.interpolate import interp1d

planets = ['Uranus']

df = pd.read_csv('../data/uranus-high-energy.csv')
df["Date"] = pd.to_datetime(df["Lcdate"], format="%m/%d/%Y")
df["Path"] = df.Path

path_label_map = {'[399 699 799] ': 'ESU',
                  '[399 299 599 799]': 'EVJU',
                  '[399 399 599 799]': 'EEJU',
                  '[399 599 799] ': 'EJU'}

launcher_list = ['falcon-heavy-expendable',
				 'falcon-heavy-expendable-w-star-48',
                 'falcon-heavy-reusable',
                 'delta-IVH',
				 'delta-IVH-w-star-48',
				 'atlas-v401',
				 'atlas-v551',
                 'atlas-v551-w-star-48',
				 'vulcan-centaur-w-6-solids',
				 'vulcan-centaur-w-6-solids-w-star-48',
				 'sls-block-1',
				 'sls-block-1B',
				 'sls-block-1B-with-kick']

launcher_label_map = {
				 'falcon-heavy-expendable': 'Falcon Heavy Expendable',
				 'falcon-heavy-expendable-w-star-48': 'Falcon Heavy Expendable with STAR48',
                 'falcon-heavy-reusable': 'Falcon Heavy Reusable',
				 'atlas-v401': 'Atlas V401',
				 'atlas-v551': 'Atlas V551',
                 'delta-IVH': 'Delta IV Heavy',
				 'delta-IVH-w-star-48': 'Delta IV Heavy with STAR48',
                 'atlas-v551-w-star-48': 'Atlas V551 with STAR48',
				 'vulcan-centaur-w-6-solids': 'Vulcan Cenatur with 6 solids',
				 'vulcan-centaur-w-6-solids-w-star-48': 'Vulcan Centaur with 6 solids + STAR 48 ',
				 'sls-block-1': 'SLS Block 1',
				 'sls-block-1B': 'SLS Block 1B',
				 'sls-block-1B-with-kick': 'SLS Block 1B with kick stage'}

def get_path_labels(row):
	if row['Path'] == '[399 699 799] ':
		return 'ESU'
	elif row['Path'] == '[399 299 599 799]':
		return 'EVJU'
	elif row['Path'] == '[399 399 599 799]':
		return 'EEJU'
	elif row['Path'] == '[399 599 799] ':
		return 'EJU'

df["Gravity Assist Path"] = df.apply(lambda row: get_path_labels(row), axis=1)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
server = app.server

main_layout = html.Div([
	dbc.Col([
		html.H1('JPL-Purdue Planetary Mission Design Suite'),
	], style={'textAlign': 'center'}),
	html.Div([
		dbc.NavbarSimple([
			dbc.DropdownMenu([
				dbc.DropdownMenuItem(planet, href=planet) for planet in planets], label='Search Trajectories')
		], brand='Home', brand_href='/'),
		dcc.Location(id='main_location'),
		html.Div(id='main_content')
	])
])

planet_layout = html.Div([
	dbc.Row([
	dbc.Col([
			html.H3(html.Strong('Search Trajectories to Uranus')),
			], style={'textAlign': 'center', 'fontWeight': 'bold'}),
		]),
	
	html.Br(),
	
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
				html.Div(children="Select Launch Window"),
				dcc.DatePickerRange(
				id="launch-window-date-picker",
				min_date_allowed=df.Date.min().date(),
				max_date_allowed=df.Date.max().date(),
				start_date=df.Date.min().date(),
				end_date=df.Date.max().date(),
			),
		], lg=3),
		dbc.Col([
			html.Div(children="Select Gravity Assist Paths"),
			dcc.Dropdown(id='gravity-assist-path-dropdown',
						 options=[{'label': path_label_map[path], 'value': path} for path in df.Path.unique()],
			             value=['[399 599 799] ', '[399 299 599 799]', '[399 399 599 799]', '[399 699 799] ' ],
			             multi=True,
			             clearable=False)
		], lg=3),
		dbc.Col([
			html.Div(children='Select Max. C3'),
			dcc.Dropdown(id='max-C3-dropdown',
			             options=[{'label': C3, 'value': C3} for C3 in np.arange(5, df.LC3.max()+1, step=10)],
			             value=185,
			             clearable=False)
		], lg=1),
		dbc.Col([
			html.Div(children='Select Max. ToF'),
			dcc.Dropdown(id='max-tof-dropdown',
			             options=[{'label': TOF, 'value': TOF} for TOF in np.arange(4, df.TOF.max()+1, step=1)],
			             value=10,
			             clearable=False)
		], lg=1),
		dbc.Col([
			html.Div(children='Select Max. Vinf'),
			dcc.Dropdown(id='max-vinf-dropdown',
			             options=[{'label': vinf, 'value': vinf} for vinf in np.arange(df.Avinf.min(), df.Avinf.max()+1, step=1)],
			             value=20,
			             clearable=False)
		], lg=1),
	dbc.Col(lg=2),
	]),
	
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dcc.Loading([
				dcc.Graph(id='c3-vs-launch-window')
			]),
		], lg=10),
		dbc.Col(lg=1)
	]),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dcc.Loading([
				dcc.Graph(id='tof-vs-launch-window')
			]),
		], lg=10),
		dbc.Col(lg=1)
	]),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dcc.Loading([
				dcc.Graph(id='avinf-vs-launch-window')
			]),
		], lg=10),
		dbc.Col(lg=1)
	]),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dcc.Loading([
				dcc.Graph(id='c3-vs-tof')
			])
		], lg=5),
		dbc.Col([
			dcc.Loading([
				dcc.Graph(id='avinf-vs-tof')
			])
		], lg=5),
		dbc.Col(lg=1)
	]),
	dbc.Row([
		dbc.Col([
			html.H3(html.Strong('Evaluate Launch Vehicle Performance')),
			], style={'textAlign': 'center'}),
	]),
	html.Br(),
	dbc.Row([
		dbc.Col(lg=4),
		dbc.Col([
			html.Div(children="Select Launch Vehicle"),
			dcc.Dropdown(id='launch-vehicle-dropdown',
			             options=[{'label': launcher_label_map[launcher], 'value': launcher} for launcher in launcher_list],
			             value='falcon-heavy-expendable',
			             clearable=False)
		], lg=4),
		dbc.Col(lg=4)
	]),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dcc.Loading([
				dcc.Graph(id='launch-mass-vs-launch-window')
			])
		], lg=10),
		dbc.Col(lg=1)
	]),
dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dcc.Loading([
				dcc.Graph(id='launch-mass-vs-tof')
			])
		], lg=5),
		dbc.Col([
			dcc.Loading([
				dcc.Graph(id='launch-mass-vs-avinf')
			])
		], lg=5),
		dbc.Col(lg=1)
	]),
	
	
	])
	

home_layout = html.Div([
	dbc.Col([
			html.H3('Select a planet on the right to get started'),
		], style={'textAlign': 'center'}),
	
	html.Br(),
	
	html.Footer([
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dbc.Tabs([
				dbc.Tab([
					html.H6("This work was performed at Purdue University under contract to the Jet Propulsion Laboratory, California Institute of Technology.")
				], label='Funding Acknowledgement'),
				dbc.Tab([
					html.H6("Under Construction")
				], label='Project Info')
			]),
		])
	])
		], style={'position': 'absolute', 'bottom': 100, 'width': 1000})
])

app.validation_layout = html.Div([
	main_layout,
	planet_layout,
	home_layout
])

app.layout = main_layout





@app.callback(Output('main_content', 'children'),
			  Input('main_location', 'pathname'))
def display_content(pathname):
	if unquote(pathname[1:]) in planets:
		return planet_layout
	else:
		return home_layout
	

@app.callback(Output('c3-vs-launch-window', 'figure'),
				Output('tof-vs-launch-window', 'figure'),
				Output('avinf-vs-launch-window', 'figure'),
				Output('c3-vs-tof', 'figure'),
				Output('avinf-vs-tof', 'figure'),
				Input('launch-window-date-picker', 'start_date'),
				Input('launch-window-date-picker', 'end_date'),
                Input('gravity-assist-path-dropdown', 'value'),
                Input('max-C3-dropdown', 'value'),
                Input('max-tof-dropdown', 'value'),
                Input('max-vinf-dropdown', 'value'))
def trajectory_trade_space_charts(start_date, end_date, paths, max_C3, max_tof, max_vinf):
	mask = ((df.Date >= start_date) &
			(df.Date <= end_date) &
			(df['Path'].isin(paths)) &
			(df['LC3'] <= max_C3) &
			(df['TOF'] <= max_tof) &
			(df['Avinf'] <= max_vinf))
	df_1 = df.loc[mask, :]
	
	c3_fig = px.scatter(df_1, x='Date', y='LC3',
	                 color='Gravity Assist Path',
	                 hover_name='Gravity Assist Path',
	                 title="C3 vs. Launch Date",
	                 height=700)
	
	c3_fig.update_layout(xaxis_title="Launch Date",
	                  yaxis_title="Launch C3",
	                  font=dict(size=13))
	
	tof_fig = px.scatter(df_1, x='Date', y='TOF',
	                    color='Gravity Assist Path',
	                    hover_name='Gravity Assist Path',
	                    title="TOF vs. Launch Date",
	                    height=700)
	
	tof_fig.update_layout(xaxis_title="Launch Date",
	                     yaxis_title="ToF, years",
	                     font=dict(size=13))
	
	avinf_fig = px.scatter(df_1, x='Date', y='Avinf',
	                     color='Gravity Assist Path',
	                     hover_name='Gravity Assist Path',
	                     title="Arrival Vinf vs. Launch Date",
	                     height=700)
	
	avinf_fig.update_layout(xaxis_title="Launch Date",
	                      yaxis_title="Vinf, km/s",
	                      font=dict(size=13))
	
	c3_tof_fig = px.scatter(df_1, x='TOF', y='LC3',
	                    color='Gravity Assist Path',
	                    hover_name='Gravity Assist Path',
	                    title="C3 vs. TOF",
	                    height=700)
	
	c3_tof_fig.update_layout(xaxis_title="TOF, years",
	                     yaxis_title="Launch C3",
	                     font=dict(size=13))
	
	avinf_tof_fig = px.scatter(df_1, x='TOF', y='Avinf',
	                        color='Gravity Assist Path',
	                        hover_name='Gravity Assist Path',
	                        title="Avinf vs. TOF",
	                        height=700)
	
	avinf_tof_fig.update_layout(xaxis_title="TOF, years",
	                         yaxis_title="Vinf, km/s",
	                         font=dict(size=13))
	
	
	return c3_fig, tof_fig, avinf_fig, c3_tof_fig, avinf_tof_fig


@app.callback(Output('launch-mass-vs-launch-window', 'figure'),
				Output('launch-mass-vs-tof', 'figure'),
				Output('launch-mass-vs-avinf', 'figure'),
				Input('launch-window-date-picker', 'start_date'),
				Input('launch-window-date-picker', 'end_date'),
                Input('gravity-assist-path-dropdown', 'value'),
                Input('max-C3-dropdown', 'value'),
                Input('max-tof-dropdown', 'value'),
                Input('max-vinf-dropdown', 'value'),
                Input('launch-vehicle-dropdown', 'value'))
def launch_mass_capability_chart(start_date, end_date, paths, max_C3, max_tof, max_vinf, launcher):

	XY = np.loadtxt(f"../data/{launcher}.csv", delimiter=',')
	f = interp1d(XY[:, 0], XY[:, 1], kind='linear', fill_value=0, bounds_error=False)
	
	mask = ((df.Date >= start_date) &
	        (df.Date <= end_date) &
	        (df['Path'].isin(paths)) &
	        (df['LC3'] <= max_C3) &
	        (df['TOF'] <= max_tof) &
	        (df['Avinf'] <= max_vinf) &
	         f(df["LC3"]) > 0)
	df_1 = df.loc[mask, :]
	
	
	launch_mass_fig = px.scatter(df_1, x='Date', y=f(df_1["LC3"]),
	                             color='Gravity Assist Path',
	                             hover_name='Gravity Assist Path',
	                             title="Launch Capability vs. Launch Date",
	                             height=700)
	
	launch_mass_fig.update_layout(xaxis_title="Launch Date",
	                              yaxis_title="Launch Capability, kg",
	                              font=dict(size=13))
	
	launch_mass_vs_tof_fig = px.scatter(df_1, x='TOF', y=f(df_1["LC3"]),
	                             color='Gravity Assist Path',
	                             hover_name='Gravity Assist Path',
	                             title="Launch Capability vs. TOF",
	                             height=700)
	
	launch_mass_vs_tof_fig.update_layout(xaxis_title="TOF, years",
	                              yaxis_title="Launch Capability ,kg",
	                              font=dict(size=13))
	
	launch_mass_vs_avinf_fig = px.scatter(df_1, x='Avinf', y=f(df_1["LC3"]),
	                                    color='Gravity Assist Path',
	                                    hover_name='Gravity Assist Path',
	                                    title="Launch Capability vs. Arrival Vinf",
	                                    height=700)
	
	launch_mass_vs_avinf_fig.update_layout(xaxis_title="Vinf, km/s",
	                                     yaxis_title="Launch Capability, kg",
	                                     font=dict(size=13))
	
	return launch_mass_fig, launch_mass_vs_tof_fig, launch_mass_vs_avinf_fig
	

if __name__ == '__main__':
	app.run_server(debug=False)
