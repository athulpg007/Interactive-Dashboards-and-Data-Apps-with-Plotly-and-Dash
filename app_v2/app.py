import datetime

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash_table import DataTable
from dash.exceptions import PreventUpdate
import plotly.express as px
import numpy as np
import pandas as pd
from urllib.parse import unquote
from scipy.interpolate import interp1d

# List of planetary destinations
planets = ['Uranus', 'Neptune']

# Dictionary with interplanetary trajectory data files and Lcdate_format
data_files = {'Uranus': {'files': ['../data/uranus-high-energy.csv',
                                   '../data/uranus-chem.csv',
								   '../data/uranus-sep.csv',
                                   '../data/uranus-sep-atlas-v551.csv',
                                   '../data/uranus-sep-delta-IVH.csv',
                                   '../data/uranus-sep-sls.csv'],
                         'Lcdate_format': ['%m/%d/%Y',
                                           '%Y%m%d',
                                           '%Y%m%d',
                                           '%Y%m%d',
                                           '%Y%m%d',
                                           '%Y%m%d']},
              'Neptune': {'files': ['../data/neptune-high-energy.csv',
                                    '../data/neptune-chem.csv',
                                    '../data/neptune-sep.csv',
									'../data/neptune-sep-atlas-v551.csv',
									'../data/neptune-sep-delta-IVH.csv',
									'../data/neptune-sep-sls.csv'
                                   ],
                         'Lcdate_format': ['%Y-%m-%d',
										   '%Y%m%d',
										   '%Y%m%d',
										   '%Y%m%d',
										   '%Y%m%d',
										   '%Y%m%d'
                                           ]}
              }

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


# Initialize empty dictionary for dataframes list of interplanetary trajectory data
df_dict = {planet: [] for planet in planets}


def get_gravity_assist_path_label(path):
	path = path.strip()
	planet_map = {'2': 'V', '3': 'E', '4': 'M', '5': 'J', '6': 'S', '7': 'U', '8': 'N'}
	label = ''
	for num in path:
		label = label + (planet_map[num])
	return label

def get_path_labels(row):
	label = get_gravity_assist_path_label(str(row['Path']))
	return label

# Populate dataframes list with trajectory data
for planet in planets:
	for file, Lcdate_format in zip(data_files[planet]['files'], data_files[planet]['Lcdate_format']):
		df = pd.read_csv(file)
		df["Date"] = pd.to_datetime(df["Lcdate"], format=Lcdate_format)
		df["Gravity Assist Path"] = df.apply(lambda row: get_path_labels(row), axis=1)
		df_dict[planet].append(df)

		
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
	dcc.Location(id='planet_page_location'),
	dbc.Row([
		dbc.Col([
			html.Div(id='planet-page-heading'),
		], style={'textAlign': 'center', 'fontWeight': 'bold'}),
	]),
	
	html.Br(),
	
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			html.Div(children="Select Launch Window"),
			dcc.DatePickerRange(id="launch-window-date-picker"),
		], lg=3),
		dbc.Col([
			html.Div(children='Select Max. C3'),
			dcc.Dropdown(id='max-C3-dropdown', clearable=False)
		], lg=1),
		dbc.Col([
			html.Div(children='Select Max. ToF'),
			dcc.Dropdown(id='max-tof-dropdown', clearable=False)
		], lg=1),
		dbc.Col([
			html.Div(children='Select Max. Vinf'),
			dcc.Dropdown(id='max-Avinf-dropdown', clearable=False)
		], lg=1),
	
	]),
	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			html.Div(children="Select Gravity Assist Paths"),
			dcc.Dropdown(id='gravity-assist-path-dropdown', multi=True, clearable=False)
				], lg=9),
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
			dcc.Dropdown(id='launch-vehicle-dropdown', clearable=False)
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
	dbc.Row([
		dbc.Col([
			html.H3(html.Strong('Trajectory Search Results')),
			], style={'textAlign': 'center'}),
	]),
	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dcc.Loading([
				html.Div(id='trajectory-search-results-table')
			])
		], lg=9),
		dbc.Col(lg=2)
	])
	
	
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
						html.H6(
							"This work was performed at Purdue University under contract to the Jet Propulsion Laboratory, California Institute of Technology.")
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


@app.callback(Output('planet-page-heading', 'children'),
              Input('planet_page_location', 'pathname'))
def get_planet_page_title(pathname):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
		return html.H3(html.Strong(f'Search Trajectories to {planet}'))


@app.callback(Output('launch-window-date-picker', 'start_date'),
			  Output('launch-window-date-picker', 'end_date'),
			  Output('launch-window-date-picker', 'min_date_allowed'),
			  Output('launch-window-date-picker', 'max_date_allowed'),
              Input('planet_page_location', 'pathname'))
def update_datepicker(pathname):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
	
	df = pd.concat(df_dict[planet])
	return df.Date.min().date(), df.Date.max().date(), df.Date.min().date(), df.Date.max().date()
	
	
@app.callback(Output('gravity-assist-path-dropdown', 'options'),
			  Output('gravity-assist-path-dropdown', 'value'),
              Input('planet_page_location', 'pathname'))
def update_path_dropdown(pathname):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
	
	df = pd.concat(df_dict[planet])
	path_options = [{'label': path, 'value': path} for path in df["Gravity Assist Path"].unique()]
	path_values = df["Gravity Assist Path"].unique()
	return path_options, path_values
	

@app.callback(Output('max-C3-dropdown', 'options'),
			  Output('max-C3-dropdown', 'value'),
              Input('planet_page_location', 'pathname'))
def update_C3_dropdown(pathname):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
	
	df = pd.concat(df_dict[planet])
	C3_options = [{'label': C3, 'value': C3} for C3 in np.arange(5, np.ceil(df.LC3.max())+1, step=5)]
	C3_value = df.LC3.max()
	return C3_options, C3_value


@app.callback(Output('max-tof-dropdown', 'options'),
              Output('max-tof-dropdown', 'value'),
              Input('planet_page_location', 'pathname'))
def update_TOF_dropdown(pathname):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
	
	df = pd.concat(df_dict[planet])
	TOF_options = [{'label': TOF, 'value': TOF} for TOF in np.arange(np.ceil(df.TOF.min()), np.ceil(df.TOF.max())+1, step=1)]
	TOF_value = np.ceil(df.TOF.max())
	return TOF_options, TOF_value


@app.callback(Output('max-Avinf-dropdown', 'options'),
              Output('max-Avinf-dropdown', 'value'),
              Input('planet_page_location', 'pathname'))
def update_Avinf_dropdown(pathname):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
	
	df = pd.concat(df_dict[planet])
	Avinf_options = [{'label': Avinf, 'value': Avinf} for Avinf in np.arange(np.ceil(df.Avinf.min()), np.ceil(df.Avinf.max()) + 1, step=1)]
	Avinf_value = np.ceil(df.Avinf.max())
	return Avinf_options, Avinf_value


@app.callback(Output('launch-vehicle-dropdown', 'options'),
              Output('launch-vehicle-dropdown', 'value'),
              Input('planet_page_location', 'pathname'))
def update_Avinf_dropdown(pathname):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
	
	df = pd.concat(df_dict[planet])
	launcher_options = [{'label': launcher_label_map[launcher], 'value': launcher} for launcher in launcher_list]
	launcher_value = 'falcon-heavy-expendable'
	return launcher_options, launcher_value


@app.callback(Output('c3-vs-launch-window', 'figure'),
			  Output('tof-vs-launch-window', 'figure'),
			  Output('avinf-vs-launch-window', 'figure'),
			  Output('c3-vs-tof', 'figure'),
			  Output('avinf-vs-tof', 'figure'),
              Output('trajectory-search-results-table', 'children'),
			  Input('planet_page_location', 'pathname'),
              Input('launch-window-date-picker', 'start_date'),
              Input('launch-window-date-picker', 'end_date'),
              Input('gravity-assist-path-dropdown', 'value'),
              Input('max-C3-dropdown', 'value'),
              Input('max-tof-dropdown', 'value'),
              Input('max-Avinf-dropdown', 'value'))
def trajectory_trade_space_charts(pathname, start_date, end_date, paths, maxC3, maxTOF, max_Avinf):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
		
	df = pd.concat(df_dict[planet])
	
	mask = ((df.Date >= start_date) &
	        (df.Date <= end_date) &
	        (df["Gravity Assist Path"].isin(paths)) &
	        (df['LC3'] <= maxC3) &
	        (df['TOF'] <= maxTOF) &
	        (df['Avinf'] <= max_Avinf))
	
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
	
	table = DataTable(columns=[{'name': col, 'id': col} for col in df_1.columns],
	                  data=df_1.to_dict('records'),
	                  style_header={'whiteSpace': 'normal'},
	                  fixed_rows={'headers': True},
	                  virtualization=True,
	                  style_table={'height': '400px'},
	                  sort_action='native',
	                  filter_action='native',
	                  export_format='none',
	                  style_cell={'minWidth': '150px'},
	                  page_size=20),
	
	return c3_fig, tof_fig, avinf_fig, c3_tof_fig, avinf_tof_fig, table


@app.callback(Output('launch-mass-vs-launch-window', 'figure'),
			  Output('launch-mass-vs-tof', 'figure'),
			  Output('launch-mass-vs-avinf', 'figure'),
			  Input('planet_page_location', 'pathname'),
              Input('launch-window-date-picker', 'start_date'),
              Input('launch-window-date-picker', 'end_date'),
              Input('gravity-assist-path-dropdown', 'value'),
              Input('max-C3-dropdown', 'value'),
              Input('max-tof-dropdown', 'value'),
              Input('max-Avinf-dropdown', 'value'),
              Input('launch-vehicle-dropdown', 'value'))
def launch_mass_capability_chart(pathname, start_date, end_date, paths, maxC3, maxTOF, max_Avinf, launcher):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
	
	df = pd.concat(df_dict[planet])
	
	XY = np.loadtxt(f"../data/{launcher}.csv", delimiter=',')
	f = interp1d(XY[:, 0], XY[:, 1], kind='linear', fill_value=0, bounds_error=False)
	
	mask = ((df.Date >= start_date) &
	        (df.Date <= end_date) &
	        (df["Gravity Assist Path"].isin(paths)) &
	        (df['LC3'] <= maxC3) &
	        (df['TOF'] <= maxTOF) &
	        (df['Avinf'] <= max_Avinf) &
	        (f(df["LC3"])) > 0)
	
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
	

# for planet in planets:
# 	for df in df_dict[planet]:
# 		print(df.head())

# df = pd.concat(df_dict['Uranus'])
# print(df)
#
if __name__ == '__main__':
	app.run_server(debug=False)
