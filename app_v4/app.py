import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash_table import DataTable
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from urllib.parse import unquote
from scipy.interpolate import interp1d

# List of planetary destinations we have data for
planets = ['Uranus', 'Neptune']

# Planet labels, name and label mappings
planet_labels = ['V', 'E', 'M', 'J', 'S', 'U', 'N']
planet_name_map = {'V': 'Venus', 'E': 'Earth', 'M': 'Mars', 'J': 'Jupiter', 'S': 'Saturn', 'U': 'Uranus', 'N': 'Neptune'}
planet_label_map = {'Venus': 'V', 'Earth': 'E', 'Mars': 'M', 'Jupiter': 'J', 'Saturn': 'S', 'Uranus': 'U', 'Neptune': 'N'}

# Trajectory Classes
trajectory_classes = ['Ballistic', 'Solar Electric Propulsion']

# Dictionary with interplanetary trajectory data filenames and Lcdate_format
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

# Launch Vehicle list
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

# Launch Vehicle label and name map
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

# Initialize empty dictionary for dataframes list of launcher C3 data
launcher_dict = {launcher: [] for launcher in launcher_list}

def get_ordered_flyby_bodies(list):
	"""
	:param list: list
		unordered list of flyby bodies
	:return: ordered_list : list
		ordered list of flyby bodies
	"""
	ordered_list=[]
	for label in planet_labels:
		if label in list:
			ordered_list.append(label)
	return ordered_list

def get_gravity_assist_path_label(path):
	"""
	:param path: str
		Gravity assist path in the format 335 for Earth-Earth-Jupiter
	:return: label : str
		Gravity assist path label in the format 'EEJ'
	"""
	path = path.strip()
	planet_map = {'2': 'V', '3': 'E', '4': 'M', '5': 'J', '6': 'S', '7': 'U', '8': 'N'}
	label = ''
	for num in path:
		label = label + (planet_map[num])
	return label

def get_path_labels(row):
	label = get_gravity_assist_path_label(str(row['Path']))
	return label

def make_empty_fig():
	"""
	:return: go.Figure()
		empty placeholder figure
	"""
	fig = go.Figure()
	fig.update_layout(
		xaxis={"visible": True},
		yaxis={"visible": True},
		annotations=[
			{
				"text": "No Data Available",
				"xref": "paper",
				"yref": "paper",
				"showarrow": False,
				"font": {
					"size": 28
				}
			}
		]
	)
	return fig

def make_empty_table(df):
	"""
	:param df: pd.DataFrame
		dataframe with column names
	:return: DataTable
		empty placeholder table
	"""
	table = DataTable(columns=[{'name': col, 'id': col} for col in df.columns[0:7]],
	                  style_header={'whiteSpace': 'normal'},
	                  fixed_rows={'headers': True},
	                  virtualization=True,
	                  style_table={'height': '400px'},
	                  sort_action='native',
	                  filter_action='native',
	                  export_format='none',
	                  style_cell={'minWidth': '150px'},
	                  page_size=10)
	
	return table

def no_trajectory_message(status):
	"""
	:param status: bool
		alert status
	:return:
	"""
	if status == True:
		return dbc.Alert("Oops! We could not find any trajectories that satisfy the search filters \
						  you selected above. Please try relaxing your search criteria.", color="danger")
	else:
		return html.H6('')
	
def no_launch_mass_message(status):
	"""
	:param status: bool
		alert status
	:return:
	"""
	if status == True:
		return dbc.Alert("Oops! No trajectories in the search results have a launch capability \
						  with the launch vehicle you selected. Please try a different launch vehicle \
						  or relaxing your search criteria.", color="danger")
	else:
		return html.H6('')
		

# Populate dataframes list with trajectory data for all planets, all data files for each planet
for planet in planets:
	for file, Lcdate_format in zip(data_files[planet]['files'], data_files[planet]['Lcdate_format']):
		df = pd.read_csv(file)
		df["Date"] = pd.to_datetime(df["Lcdate"], format=Lcdate_format)
		df["Gravity Assist Path"] = df.apply(lambda row: get_path_labels(row), axis=1)
		df["Flyby Bodies"] = df.apply(lambda row: row["Gravity Assist Path"][1:-1], axis=1)
		if 'P0' in df.columns:
			df["Type"] = "Solar Electric Propulsion"
		else:
			df["Type"] = "Ballistic"
		
		df_dict[planet].append(df)

df_launchers = pd.DataFrame(columns=['id', "Launch Vehicle", "C3", "Launch Capability, kg"])


for launcher in launcher_list:
	XY = np.loadtxt(f"../data/{launcher}.csv", delimiter=',')
	f = interp1d(XY[:, 0], XY[:, 1], kind='linear', fill_value=0, bounds_error=False)
	launcher_dict[launcher] = [XY[0, 0], XY[-1, 0], f]

	for C3 in np.linspace(XY[0, 0], XY[-1, 0], 41):
		entry = pd.DataFrame([{
			'id': launcher,
			"Launch Vehicle": launcher_label_map[launcher],
			"C3": C3,
			"Launch Capability, kg": f(C3)
		}])

		df_launchers = pd.concat([df_launchers, entry], ignore_index=True)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
server = app.server

main_layout = html.Div([
	dbc.Col([
		html.H1('JPL-Purdue Planetary Mission Design Suite'),
	], style={'textAlign': 'center'}),
	html.Div([
		dbc.NavbarSimple([
			dbc.Row([
				dbc.Col([
					dbc.DropdownMenu([
						dbc.DropdownMenuItem(planet, href=planet) for planet in planets], label='Search Trajectories'),
				]),
				dbc.Col([
					dbc.Button('Launch Performance', href='/launch_perf')
				]),
			]),
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
			html.Div(children=html.Strong("Select Launch Window")),
			dcc.DatePickerRange(id="launch-window-date-picker"),
		], lg=3),
		dbc.Col([
			html.Div(children=html.Strong('Select Max. C3')),
			dcc.Dropdown(id='max-C3-dropdown', clearable=False)
		], lg=1),
		dbc.Col([
			html.Div(children=html.Strong('Select Max. ToF')),
			dcc.Dropdown(id='max-tof-dropdown', clearable=False)
		], lg=1),
		dbc.Col([
			html.Div(children=html.Strong('Select Max. Vinf')),
			dcc.Dropdown(id='max-Avinf-dropdown', clearable=False)
		], lg=1),
		dbc.Col([
			html.Div(children=html.Strong('Max. DSM DV')),
			dcc.Dropdown(id='max-dsm-dv-dropdown', clearable=False)
		], lg=1),
	
	]),
	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			html.Div(children=html.Strong('Select Gravity Assist Flyby Bodies:'))
		], lg=2),
		dbc.Col([
			dcc.Checklist(id="flyby-bodies-checkbox",
			              inputStyle={"margin-right": "5px"},
			              labelStyle={'display': 'inline-block', "margin-left": "10px"})
		], lg=3),
		dbc.Col([
			html.Div(children=html.Strong('Select Trajectory Type:'))
		], lg=1.5),
		dbc.Col([
			dcc.Checklist(id="trajectory-type-checkbox",
			              options=[{'label': traj_class, 'value': traj_class} for traj_class in trajectory_classes],
			              value=trajectory_classes,
			              inputStyle={"margin-right": "5px"},
			              labelStyle={'display': 'inline-block', "margin-left": "10px"})
		], lg=3),
		dbc.Col(lg=1),
	]),
	
	dbc.Row([
		dbc.Col(lg=3),
		dbc.Col(html.Div(id='no-data-error-message'), lg=6),
		dbc.Col(lg=3)
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
			html.Div(children=html.Strong("Select Launch Vehicle")),
			dcc.Dropdown(id='launch-vehicle-dropdown', clearable=False)
		], lg=4),
		dbc.Col(lg=4)
	]),
	dbc.Row([
		dbc.Col(lg=2),
		dbc.Col(html.Div(id='no-launch-mass-message'), lg=8),
		dbc.Col(lg=2)
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

launcher_layout = html.Div([
	dcc.Location(id='launch_page_location'),

	dbc.Col([
		html.Div(id='launch-page-heading'),
	], style={'textAlign': 'center'}),

	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col(html.H4('Available Launch Options'), lg=4),
		dbc.Col(lg=7)
	]),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col(dcc.Checklist(id="launch-vehicles-checkbox",
			              inputStyle={"margin-right": "5px"},
			              labelStyle={'display': 'inline-block', "margin-left": "20px"}), lg=10),
		dbc.Col(lg=1)
	]),
	dbc.Row([
			dbc.Col(lg=1),
			dbc.Col([
				dcc.Loading([
					dcc.Graph(id='launch-mass-vs-c3-chart')
				]),
			], lg=10),
			dbc.Col(lg=1)
		]),
	dbc.Row([
		dbc.Col(lg=2),
		dbc.Col([html.Div(children=html.Strong('Select a launcher')),
			dcc.Dropdown(id='launch-vehicle-dropdown-1', clearable=False)], lg=4),
		dbc.Col([html.Div(children=html.Strong('Enter C3')),
				dcc.Input(id='launch-c3-value', value=10, min=0, max=225)], lg=2),
		dbc.Col([html.Div(children=html.Strong('Launch capability')),
				html.Div(id='launch-capability-value', style={'color': 'blue'}), ], lg=4),
		dbc.Col(lg=1)
	]),
	html.Br(),
	dbc.Row([
		dbc.Col(lg=2),
		dbc.Col(html.Div(html.H6("This work was performed at Purdue University under contract \
								 to the Jet Propulsion Laboratory, California Institute of Technology.")), lg=8),
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
						html.H6("This work was performed at Purdue University under contract \
								 to the Jet Propulsion Laboratory, California Institute of Technology.")
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
	launcher_layout,
	home_layout,
])

app.layout = main_layout


@app.callback(Output('main_content', 'children'),
              Input('main_location', 'pathname'))
def display_content(pathname):
	if unquote(pathname[1:]) in planets:
		return planet_layout
	if unquote(pathname[1:]) == 'launch_perf':
		return launcher_layout
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


@app.callback(Output('launch-page-heading', 'children'),
              Input('launch_page_location', 'pathname'))
def get_launch_page_title(pathname):
	if unquote(pathname[1:]) in 'launch_perf':
		return html.H3(html.Strong(f'Launch Vehicle Performance Calculator'))


@app.callback(Output('launch-mass-vs-c3-chart', 'figure'),
              Input('launch_page_location', 'pathname'),
			  Input('launch-vehicles-checkbox', 'value'),)
def get_launch_page_main_chart(pathname, launchers):
	if unquote(pathname[1:]) in 'launch_perf':
		df = df_launchers[df_launchers['id'].isin(launchers)]
		fig = px.line(df, x='C3', y='Launch Capability, kg', color='Launch Vehicle', markers=True)
		fig.update_layout(height=600)
		return fig


@app.callback(Output('launch-capability-value', 'children'),
              Input('launch-vehicle-dropdown-1', 'value'),
			  Input('launch-c3-value', 'value'))
def get_launch_capability(launcher, C3):
	f = launcher_dict[launcher][2]
	if C3:
		ans = float(f(C3))
		return html.H4(html.Strong(f'{round(ans)} kg'), )
	else:
		return None



@app.callback(Output('launch-vehicle-dropdown-1', 'options'),
			  Output('launch-vehicle-dropdown-1', 'value'),
			  Input('launch_page_location', 'pathname'))
def update_launcher_dropdown(pathname):
	launch_options = [{'label': launcher_label_map[launcher], 'value': launcher} for launcher in launcher_list]
	launch_value = launch_options[0]['value']
	return launch_options, launch_value


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
	
	
@app.callback(Output('flyby-bodies-checkbox', 'options'),
			  Output('flyby-bodies-checkbox', 'value'),
              Input('planet_page_location', 'pathname'))
def update_flyby_bodies_checkboxes(pathname):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
	
	df = pd.concat(df_dict[planet])
	flyby_bodies_set = set(''.join(df["Flyby Bodies"]))
	flyby_bodies_list = list(flyby_bodies_set)
	flyby_bodies_list = get_ordered_flyby_bodies(flyby_bodies_list)
	flyby_options = [{'label': planet_name_map[body], 'value': body} for body in flyby_bodies_list]
	flyby_values = [body for body in flyby_bodies_list]
	return flyby_options, flyby_values


@app.callback(Output('launch-vehicles-checkbox', 'options'),
			  Output('launch-vehicles-checkbox', 'value'),
			  Input('launch_page_location', 'pathname'))
def update_launch_vehicle_checkboxes(pathname):
	launch_options = [{'label': launcher_label_map[launcher], 'value': launcher} for launcher in launcher_list]
	launch_values = [launcher for launcher in launcher_list]
	return launch_options, launch_values
	

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

@app.callback(Output('max-dsm-dv-dropdown', 'options'),
              Output('max-dsm-dv-dropdown', 'value'),
              Input('planet_page_location', 'pathname'))
def update_DSMdV_dropdown(pathname):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
	
	df = pd.concat(df_dict[planet])
	DSMdv_options = [{'label': DSMdv, 'value': DSMdv} for DSMdv in
	                np.arange(np.floor(df.DSMdv.min()), np.ceil(df.DSMdv.max()) + 0.5, step=0.5)]
	DSMdv_value = 1.0
	return DSMdv_options, DSMdv_value

@app.callback(Output('launch-vehicle-dropdown', 'options'),
              Output('launch-vehicle-dropdown', 'value'),
              Input('planet_page_location', 'pathname'))
def update_launcher_dropdown(pathname):
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
			  Output('no-data-error-message', 'children'),
			  Input('planet_page_location', 'pathname'),
              Input('launch-window-date-picker', 'start_date'),
              Input('launch-window-date-picker', 'end_date'),
              Input('flyby-bodies-checkbox', 'value'),
              Input('max-C3-dropdown', 'value'),
              Input('max-tof-dropdown', 'value'),
              Input('max-Avinf-dropdown', 'value'),
              Input('max-dsm-dv-dropdown', 'value'),
              Input('trajectory-type-checkbox', 'value'))
def trajectory_trade_space_charts(pathname, start_date, end_date, flyby_bodies, maxC3, maxTOF, max_Avinf, max_DSMdv,
                                  traj_types):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
		
	df = pd.concat(df_dict[planet])
	flyby_bodies_set = [set(list(x)) for x in df["Flyby Bodies"]]
	df["Flyby Visibility"] = [x.issubset(set(flyby_bodies)) for x in flyby_bodies_set]
	df["DSMdv"] = df["DSMdv"].fillna(0)
	
	mask = ((df.Date >= start_date) &
	        (df.Date <= end_date) &
	        (df["Flyby Visibility"] == True) &
	        (df['LC3'] <= maxC3) &
	        (df['TOF'] <= maxTOF) &
	        (df['Avinf'] <= max_Avinf) &
	        (df['DSMdv'] <= max_DSMdv) &
	        (df['Type'].isin(traj_types)))
	
	df_1 = df.loc[mask, :]
	
	if df_1.empty:
		return make_empty_fig(), make_empty_fig(), make_empty_fig(), make_empty_fig(), make_empty_fig(), \
		       make_empty_table(df), no_trajectory_message(status=True)
		
	
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
	                  page_size=10),
	
	return c3_fig, tof_fig, avinf_fig, c3_tof_fig, avinf_tof_fig, table, no_trajectory_message(status=False)


@app.callback(Output('launch-mass-vs-launch-window', 'figure'),
			  Output('launch-mass-vs-tof', 'figure'),
			  Output('launch-mass-vs-avinf', 'figure'),
			  Output('no-launch-mass-message', 'children'),
			  Input('planet_page_location', 'pathname'),
              Input('launch-window-date-picker', 'start_date'),
              Input('launch-window-date-picker', 'end_date'),
              Input('flyby-bodies-checkbox', 'value'),
              Input('max-C3-dropdown', 'value'),
              Input('max-tof-dropdown', 'value'),
              Input('max-Avinf-dropdown', 'value'),
			  Input('max-dsm-dv-dropdown', 'value'),
              Input('launch-vehicle-dropdown', 'value'),
              Input('trajectory-type-checkbox', 'value'))
def launch_mass_capability_chart(pathname, start_date, end_date, flyby_bodies, maxC3, maxTOF, max_Avinf, max_DSMdv,
                                 launcher, traj_types):
	if unquote(pathname[1:]) not in planets:
		raise PreventUpdate
	if unquote(pathname[1:]) in planets:
		planet = unquote(pathname[1:])
	
	df = pd.concat(df_dict[planet])
	flyby_bodies_set = [set(list(x)) for x in df["Flyby Bodies"]]
	df["Flyby Visibility"] = [x.issubset(set(flyby_bodies)) for x in flyby_bodies_set]
	df["DSMdv"] = df["DSMdv"].fillna(0)
	
	XY = np.loadtxt(f"../data/{launcher}.csv", delimiter=',')
	f = interp1d(XY[:, 0], XY[:, 1], kind='linear', fill_value=0, bounds_error=False)
	
	mask = ((df.Date >= start_date) &
	        (df.Date <= end_date) &
	        (df["Flyby Visibility"] == True) &
	        (df['LC3'] <= maxC3) &
	        (df['TOF'] <= maxTOF) &
	        (df['Avinf'] <= max_Avinf) &
	        (df['DSMdv'] <= max_DSMdv) &
	        (f(df["LC3"]) > 0) &
	        (df['Type'].isin(traj_types)))
	
	df_1 = df.loc[mask, :]
	
	if df_1.empty:
		return make_empty_fig(), make_empty_fig(), make_empty_fig(), no_launch_mass_message(status=True)
		
	
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
	
	return launch_mass_fig, launch_mass_vs_tof_fig, launch_mass_vs_avinf_fig, no_launch_mass_message(status=False)
	

if __name__ == '__main__':
	app.run_server(debug=False)
