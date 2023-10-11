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
from vtkmodules.vtkImagingCore import vtkRTAnalyticSource
from vtkmodules.vtkFiltersSources import vtkSphereSource
import dash_vtk
from dash_vtk.utils import to_mesh_state
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from AMAT.approach import Approach
from AMAT.planet import Planet
from AMAT.orbiter import PropulsiveOrbiter
from AMAT.vehicle import Vehicle
from AMAT.telecom import Link

# List of planetary destinations we have data for
planets = ['Uranus', 'Neptune']
planets_ac = ['Venus', 'Mars', 'Uranus', 'Neptune']

# Planet labels, name and label mappings
planet_labels = ['V', 'E', 'M', 'J', 'S', 'U', 'N']
planet_name_map = {'V': 'Venus', 'E': 'Earth', 'M': 'Mars', 'J': 'Jupiter', 'S': 'Saturn', 'U': 'Uranus',
				   'N': 'Neptune'}
planet_label_map = {'Venus': 'V', 'Earth': 'E', 'Mars': 'M', 'Jupiter': 'J', 'Saturn': 'S', 'Uranus': 'U',
					'Neptune': 'N'}

planet_color_map = {
	'Uranus': (0.10, 0.55, 0.35),
	'Neptune': (0, 0, 1)
}

planet_atmdata_map = {
	'Venus': '../data/venus-gram-avg.dat',
	'Mars': '../data/mars-gram-avg.dat',
	'Uranus': '../data/uranus-gram-avg.dat',
	'Neptune': '../data/neptune-gram-avg.dat'
}

planet_h_col_map = {
	'Mars': 0,
	'Venus': 0,
	'Uranus': 0,
	'Neptune': 0,
}

planet_t_col_map = {
	'Mars': 1,
	'Venus': 1,
	'Uranus': 1,
	'Neptune': 7,
}

planet_p_col_map = {
	'Mars': 2,
	'Venus': 2,
	'Uranus': 2,
	'Neptune': 6,
}

planet_rho_col_map = {
	'Mars': 3,
	'Venus': 3,
	'Uranus': 3,
	'Neptune': 5,
}

planet_h_km_flag = {
	'Mars': False,
	'Venus': False,
	'Uranus': True,
	'Neptune': True,
}

planet_entry_int_alt_km = {
	'Mars': 120,
	'Venus': 150,
	'Uranus': 1000,
	'Neptune': 1000,
}

planet_entry_speed_kms = {
	'Mars': 7,
	'Venus': 11,
	'Uranus': 28,
	'Neptune': 28,
}

planet_target_apoapsis = {
	'Mars': 400,
	'Venus': 400,
	'Uranus': 450e3,
	'Neptune': 450e3,
}

link_budget_presets = {
	'cassini-downlink': {
		'freq': 8.425, 'pt': 20, 'gt': 47.2, 'gr': 68.24, 'ts': 40.5, 'range': 1.5E9,
		'lt': 1, 'lsc': 0.2, 'la': 0.35, 'lother': 0.55, 'ldsn': 1, 'rate': 14, 'eb-n0-required': 0.31
	},
	'cassini-huygens-relay': {
		'freq': 2.040, 'pt': 11.7, 'gt': 1.05, 'gr': 34.7, 'ts': 230, 'range': 80E3,
		'lt': 1, 'lsc': 0.2, 'la': 0.05, 'lother': 3.3, 'ldsn': 0, 'rate': 8, 'eb-n0-required': 2.55
	},
	'oceanus-downlink': {
		'freq': 32, 'pt': 40, 'gt': 60.33, 'gr': 80.0, 'ts': 46.97, 'range': 3.05E9,
		'lt': 1, 'lsc': 0.2, 'la': 0.35, 'lother': 0.60, 'ldsn': 1, 'rate': 34, 'eb-n0-required': 0.31
	},
	'oceanus-saturn-probe-relay': {
		'freq': 0.405, 'pt': 10.7, 'gt': 5.10, 'gr': 22.3, 'ts': 183, 'range': 1.62E6,
		'lt': 1, 'lsc': 0.2, 'la': 2.0, 'lother': 2.75, 'ldsn': 0, 'rate': 0.2, 'eb-n0-required': 5.00
	},
	'oceanus-uranus-probe-relay': {
		'freq': 0.405, 'pt': 10.7, 'gt': 5.89, 'gr': 22.3, 'ts': 104, 'range': 1.55E5,
		'lt': 1, 'lsc': 0.2, 'la': 0.35, 'lother': 1.34, 'ldsn': 0, 'rate': 0.2, 'eb-n0-required': 5.00
	},
	'dragonfly-dte': {
		'freq': 8.425, 'pt': 30, 'gt': 30, 'gr': 80, 'ts': 40.5, 'range': 1.5E9,
		'lt': 1, 'lsc': 0.2, 'la': 0.35, 'lother': 0.54, 'ldsn': 1, 'rate': 2, 'eb-n0-required': 0.31
	}
}


# Trajectory Classes
trajectory_classes = ['Ballistic', 'Solar Electric Propulsion']

# Dictionary with interplanetary trajectory data filenames and Lcdate_format
data_files = {'Uranus': {'files': ['../data/uranus-high-energy.csv',
								   '../data/uranus-chem.csv',
								   '../data/uranus-sep.csv',
								   '../data/uranus-sep-atlas-v551.csv',
								   '../data/uranus-sep-delta-IVH.csv',
								   '../data/uranus-sep-sls.csv',
                                   '../data/EVEEJU-clean-2.csv'],
						 'Lcdate_format': ['%m/%d/%Y',
										   '%Y%m%d',
										   '%Y%m%d',
										   '%Y%m%d',
										   '%Y%m%d',
										   '%Y%m%d',
						                   '%Y-%m-%d']},
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
	ordered_list = []
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


def failed_ac_corridor(status):
	"""
	:param status: bool
		alert status
	:return:
	"""
	if status == True:
		return dbc.Alert("Failed to compute aerocapture corridor bounds.", color="danger")
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
		html.H1('Planetary Science Mission Architecture Exploration (PLASMAx)'),
	], style={'textAlign': 'center'}),
	html.Div([
		dbc.NavbarSimple([
			dbc.Row([
				dbc.Col([
					dbc.DropdownMenu([
						dbc.DropdownMenuItem(planet, href=planet) for planet in planets], label='Search Trajectories'),
				]),
				dbc.Col([
					dbc.Button('Launch', href='/launch_perf')
				]),
				dbc.Col([
					dbc.Button('Approach', href='/approach')
				]),
				dbc.Col([
					dbc.Button('Aerocapture', href='/aerocapture')
				]),
				dbc.Col([
					dbc.Button('Telecom', href='/link_budget')
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

link_budget_layout = html.Div([
	dcc.Location(id='link_budget_location'),
	dbc.Col([
		html.Div(id='link-budget-heading'),
	], style={'textAlign': 'center'}),
	html.Br(),
	dbc.Row([
	dbc.Col(lg=1),
	dbc.Col([html.Div([html.Strong('Automatic Update:'),
	        html.Abbr("\uFE56", title="Toggle to prevent update when changing input values", style={'color': 'blue', 'font-size': '18px'})]),
				 dcc.RadioItems(id='link-budget-auto-update', options = [
					 {'label': 'On', 'value': 'True'},
					 {'label': 'Off', 'value': 'False'},
				 ], value= 'True',
					inputStyle={"margin-right": "10px"},
					labelStyle={'display': 'inline-block', "margin-left": "15px"}),
			], lg=2),
	dbc.Col([html.Div([html.Strong('Presets:'),
	        html.Abbr("\uFE56", title=
	        """Example link budget presets: \n
1) Cassini Downlink to DSN from Saturn\n
2) Huygens to Cassini Relay\n
3) Oceanus Downlink from Uranus (Purdue Mission Concept Study)\n
4) Oceanus Saturn Probe to Orbiter (Purdue Mission Concept Study)\n
5) Oceanus Uranus Probe to Orbiter (Purdue Mission Concept Study)\n
6) Dragonfly Direct-to-Earth from Titan (Estimated)\n""", style={'color': 'blue', 'font-size': '18px'})]),
				 dcc.RadioItems(id='link-budget-presets', options = [
					 {'label': 'Cassini Downlink', 'value': 'cassini-downlink'},
					 {'label': 'Cassini-Huygens Relay', 'value': 'cassini-huygens-relay'},
					 {'label': 'Oceanus Downlink', 'value': 'oceanus-downlink'},
					 {'label': 'Oceanus Saturn Probe Relay', 'value': 'oceanus-saturn-probe-relay'},
					 {'label': 'Oceanus Uranus Probe Relay', 'value': 'oceanus-uranus-probe-relay'},
					 {'label': 'Dragonfly DTE', 'value': 'dragonfly-dte'},
				 ], value= 'cassini-downlink',
					inputStyle={"margin-right": "5px"},
					labelStyle={'display': 'inline-block', "margin-left": "10px"}),
			], lg=9),
	]),
	
	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([html.Div([html.Strong('Frequency, GHz'),
                 html.Abbr("\uFE56", title="Example: 2.4 GHz (S-Band), 8.425 GHz (X-Band) or 32.00 GHz (Ka-Band)", style={'color': 'blue', 'font-size': '18px'})]),
				 dcc.Input(id='link-budget-frequency', min=0.001, max=100, type='number')], lg=2),
		dbc.Col([html.Div([html.Strong('Transmit power, W'),
                html.Abbr("\uFE56", title="Transmit Power, Example: 20 W (Cassini)", style={'color': 'blue', 'font-size': '18px'})]),
				 dcc.Input(id='link-budget-pt', min=0, max=1000, type='number')], lg=2),
		dbc.Col([html.Div([html.Strong('Transmitter gain, dBi'),
		        html.Abbr("\uFE56", title="Transmitte antenna gain, Example: 47 dBi (Cassini)", style={'color': 'blue', 'font-size': '18px'})]),
				 dcc.Input(id='link-budget-gt', min=0, max=100, type='number')], lg=2),
		dbc.Col([html.Div(children=html.Strong('Receiver gain, dBi')),
				 dcc.Input(id='link-budget-gr', min=0, max=120, type='number')], lg=2),
		dbc.Col([html.Div(children=html.Strong('System Noise Temp., K')),
				 dcc.Input(id='link-budget-ts', min=0, max=300, type='number')], lg=2),
		dbc.Col(lg=1)
]),
	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([html.Div(children=html.Strong('Range, km')),
				 dcc.Input(id='link-budget-range', min=1E3, max=1.5E10, type='number')], lg=2),
		dbc.Col([html.Div(children=html.Strong('Transmitter Loss, dB')),
				 dcc.Input(id='link-budget-lt', min=0, max=10, type='number')], lg=2),
		dbc.Col([html.Div(children=html.Strong('S/C Circuit Loss, dB')),
				 dcc.Input(id='link-budget-lsc', min=0, max=1, type='number')], lg=2),
		dbc.Col([html.Div(children=html.Strong('Atmospheric loss, dB')),
				 dcc.Input(id='link-budget-la', min=0, max=5, type='number')], lg=2),
		dbc.Col([html.Div(children=html.Strong('Other losses, dB')),
				 dcc.Input(id='link-budget-lother', min=0, max=5, type='number')], lg=2),
		dbc.Col(lg=1),
	]),
	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
	dbc.Col([html.Div(children=html.Strong('DSN system loss, dB')),
				 dcc.Input(id='link-budget-ldsn', min=0, max=5, type='number')], lg=2),
	dbc.Col([html.Div(children=html.Strong('Data rate, kbps')),
				 dcc.Input(id='link-budget-rate-kbps', min=0, max=500E3, type='number')], lg=2),
	dbc.Col([html.Div(children=html.Strong('Eb/N0 required')),
				 dcc.Input(id='link-budget-eb-n0-required', min=0, max=50, type='number')], lg=2),
	dbc.Col([html.Div(children=html.Strong('Eb/N0: ')),
				 html.Div(id='link-budget-eb-n0-computed', style={'color': 'blue'}), ], lg=2),
	dbc.Col([html.Div(children=html.Strong('Eb/N0 margin: ')),
				 html.Div(id='link-budget-eb-n0-margin', style={'color': 'blue'}), ], lg=2),
	]),
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

approach_layout = html.Div([
	dcc.Location(id='approach-page-location'),

	dbc.Col([
		html.Div(id='approach-page-heading'),
	], style={'textAlign': 'center'}),

	html.Br(),

	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([html.Div(children=html.Strong('Select a planet')),
				 dcc.Dropdown(id='approach-planet-dropdown', clearable=False)], lg=2),
		dbc.Col([html.Div(children=html.Strong('Enter  V_inf vector, km/s (ICRF)')),
				 dcc.Input(id='approach-vinf-input', value='-9.625,16.511,7.464')], lg=3),
		dbc.Col([html.Div(children=html.Strong('Select approach type:')),
				 dcc.RadioItems(id='approach-is-entrysystem', options = [
					 {'label': 'Flyby', 'value': 'flyby'},
					 {'label': 'Atmospheric entry', 'value': 'probe'},
					 {'label': 'Orbiter', 'value': 'orbiter'},
				 ], value= 'probe',
								inputStyle={"margin-right": "5px"},
								labelStyle={'display': 'inline-block', "margin-left": "10px"}),
			], lg=3),
	dbc.Col([html.Div(children=html.Strong('Orbit insertion DV, m/s:')),
				 html.Div(id='orbit-insertion-dv', style={'color': 'blue'}), ], lg=3),

		dbc.Col(lg=3),
	]),

	html.Br(),

	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([html.Div(children=html.Strong('Select periapsis alt., km')),
				 html.Div(id='selected-periapsis', style={'color': 'blue'}),
				 dcc.Slider(id='approach-periapsis-slider', min=-4000, max=4000,
							marks={-4000 : {'label': '-4000 km'},
								   1000 : {'label': '+1000 km'},
								   4000: {'label': '+4000 km'}}, value=250.0,
							tooltip={"placement": "bottom"})], lg=3),
		dbc.Col([html.Div(children=html.Strong('Select B-plane clock angle., deg')),
		         html.Div(id='selected-clock-angle', style={'color': 'blue'}),
		         dcc.Slider(id='approach-psi-slider', min=0, max=360,
							marks={0 : {'label': '0 deg.'},
								   90 : {'label': '90 deg.'},
								   180: {'label': '180 deg.'},
								   270: {'label': '270 deg.'},
								   360: {'label': '360 deg.'}}, value=270.0,
							tooltip={"placement": "bottom"})], lg=3),
	dbc.Col([html.Div(children=html.Strong('Enter orbit apoapsis alt., km')),
			 html.Div(id='selected-apoapsis', style={'color': 'blue'}),
			 dcc.Slider(id='orbit-apoapsis-slider', min=4000, max=4E6,
						marks={4000: {'label': '4000 km'},
							   900000: {'label': '+900K km'},
							   2000000: {'label': '+2M km'},
							   4000000: {'label': '+4M km'}}, value=400000,
						tooltip={"placement": "bottom"})], lg=3),
		dbc.Col(lg=1),

	]),

	html.Br(),

	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dcc.Loading([
				html.Div(id='approach-main-viz',
						 style={"height": "700px"})
			]),
		], lg=10),
		dbc.Col(lg=1)
	]),
	
	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dcc.Loading([
				html.Strong('Approach Results (Entry state / orbital parameters)')
			])
		], lg=9),
		dbc.Col(lg=2)
	]),
	html.Br(),
	dbc.Row([
			dbc.Col(lg=1),
			dbc.Col([
				dcc.Loading([
					html.Div(id='approach-results-table')
				])
			], lg=9),
			dbc.Col(lg=2)
		]),
	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dcc.Loading([
				html.H3('Entry probe design parameters')
			])
		], lg=9),
		dbc.Col(lg=2)
	]),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([html.Div(children=html.Strong('Vehicle mass, kg')),
				 html.Div(id='selected-mass', style={'color': 'blue'}),
				 dcc.Slider(id='mass-slider', min=50, max=1000,
							marks={100 : {'label': '10'},
								   300 : {'label': '300'},
								   800: {'label': '800'}}, value=300.0,
							tooltip={"placement": "bottom"})], lg=3),
		dbc.Col([html.Div(children=html.Strong('Vehicle diameter, m')),
				 html.Div(id='selected-diameter', style={'color': 'blue'}),
				 dcc.Slider(id='diameter-slider', min=0.5, max=6.0, step=0.1,
							marks={1: {'label': '1.0'},
								   2 : {'label': '2.0'},
								   6: {'label': '6.0'}}, value=1.2,
							tooltip={"placement": "bottom"})], lg=3),
		dbc.Col([html.Div(children=html.Strong('Vehicle nose radius, m')),
				 html.Div(id='selected-nose-radius', style={'color': 'blue'}),
				 dcc.Slider(id='nose-radius-slider', min=0.1, max=5, step=0.1,
							marks={1 : {'label': '1.0'},
								   2: {'label': '2.0'},
								   4 : {'label': '4.0'}}, value=0.2,
							tooltip={"placement": "bottom"})], lg=3)
	]),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([html.Div(children=html.Strong('Ballistic coefficient, kg/m2: ')),
				 html.Div(id='ballistic-coefficient', style={'color': 'blue'}), ], lg=3),
	]),
	dbc.Row([
			dbc.Col(lg=1),
			dbc.Col([
				dcc.Loading([
					dcc.Graph(id='entry-trajectory-figure')
				])
			], lg=10),
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

ac_layout = html.Div([
	dcc.Location(id='ac-page-location'),

	dbc.Col([
		html.Div(id='ac-page-heading'),
	], style={'textAlign': 'center'}),
	
	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			html.Div(children=html.Strong('Select a planet')),
			dcc.Dropdown(id='ac-planet-dropdown', clearable=False)
		], lg=2),
		dbc.Col([html.Div(children=html.Strong('Select vehicle type:')),
				 dcc.RadioItems(id='ac-vehicle-type', options = [
					 {'label': 'Lift Modulation', 'value': 'lift-modulation'},
					 {'label': 'Drag Modulation', 'value': 'drag-modulation'},
				 ], value= 'lift-modulation',
					inputStyle={"margin-right": "5px"},
					labelStyle={'display': 'inline-block', "margin-left": "10px"}),
			], lg=3),
		dbc.Col([html.Div(children=html.Strong('Target apoapsis altitude, km')),
				 html.Div(id='ac-selected-apoapsis', style={'color': 'blue'}),
				 dcc.Slider(id='ac-apoapsis-slider', min=400, max=0.5E6,
							marks={400: {'label': '400'},
							       100000 : {'label': '100k'},
								   500000 : {'label': '0.5M'}},
							tooltip={"placement": "bottom"})], lg=5),
	]),
	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dcc.Loading([
				html.H3('Vehicle design parameters')
			])
		], lg=9),
		dbc.Col(lg=2)
	]),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([html.Div(children=html.Strong('Vehicle mass, kg')),
				 html.Div(id='ac-selected-mass', style={'color': 'blue'}),
				 dcc.Slider(id='ac-mass-slider', min=50, max=5000,
							marks={100 : {'label': '10'},
								   3000 : {'label': '3000'},
								   5000: {'label': '5000'}}, value=3000.0,
							tooltip={"placement": "bottom"})], lg=3),
		dbc.Col([html.Div(children=html.Strong('Vehicle diameter, m')),
				 html.Div(id='ac-selected-diameter', style={'color': 'blue'}),
				 dcc.Slider(id='ac-diameter-slider', min=0.5, max=6.0, step=0.1,
							marks={1: {'label': '1.0'},
								   2 : {'label': '2.0'},
								   6: {'label': '6.0'}}, value=4.5,
							tooltip={"placement": "bottom"})], lg=3),
		dbc.Col([html.Div(children=html.Strong('Vehicle nose radius, m')),
				 html.Div(id='ac-selected-nose-radius', style={'color': 'blue'}),
				 dcc.Slider(id='ac-nose-radius-slider', min=0.1, max=5, step=0.1,
							marks={1 : {'label': '1.0'},
								   2: {'label': '2.0'},
								   4 : {'label': '4.0'}}, value=1.0,
							tooltip={"placement": "bottom"})], lg=3)
	]),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([html.Div(children=html.Strong('Lift-to-drag ratio, L/D')),
				 html.Div(id='ac-selected-LD', style={'color': 'blue'}),
				 dcc.Slider(id='ac-LD-slider', min=0, max=1, step=0.01,
							marks={0 : {'label': '0'},
								   1 : {'label': '1'}}, value=0.36,
							tooltip={"placement": "bottom"})], lg=3),
		
		dbc.Col([html.Div(children=html.Strong('Ballistic coefficient ratio, (DM)')),
				 html.Div(id='ac-selected-BCR', style={'color': 'blue'}),
				 dcc.Slider(id='ac-bcr-slider', min=1, max=10, step=0.1,
							marks={1 : {'label': '1'},
								   10 : {'label': '10'}}, value=7.0,
							tooltip={"placement": "bottom"})], lg=3),
		
		dbc.Col([html.Div(children=html.Strong('Ballistic coefficient, kg/m2: ')),
				 html.Div(id='ac-ballistic-coefficient', style={'color': 'blue'}), ], lg=3),
		
	]),
	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([
			dcc.Loading([
				html.H3('Vehicle entry conditions')
			])
		], lg=6),
		dbc.Col(lg=2)
	]),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([html.Div(children=html.Strong('Altitude, km')),
				 dcc.Input(id='ac-entry-altitude', min=100, max=2000, readOnly=True)], lg=2),
		dbc.Col([html.Div(children=html.Strong('Longitude, deg')),
				 dcc.Input(id='ac-entry-longitude', value=0, min=0, max=360, readOnly=True)], lg=2),
		dbc.Col([html.Div(children=html.Strong('Latitude, deg')),
				 dcc.Input(id='ac-entry-latitude', value=0, min=-90, max=90, readOnly=True)], lg=2),
		dbc.Col([html.Div(children=html.Strong('Atm. rel. speed, km/s')),
				 dcc.Input(id='ac-entry-speed', min=3, max=35, type='number')], lg=2),
		dbc.Col([html.Div(children=html.Strong('Heading, deg.')),
				 dcc.Input(id='ac-entry-heading', value=0, min=0, max=360, readOnly=True)], lg=2),
		dbc.Col(lg=1)
	]),
	html.Br(),
	dbc.Row([
		dbc.Col(lg=1),
		dbc.Col([html.Div(children=html.Strong('Overshoot limit, deg: ')),
				 dcc.Loading(html.Div(id='ac-overshoot-limit', style={'color': 'blue'})), ], lg=2),
		dbc.Col([html.Div(children=html.Strong('Undershoot limit, deg: ')),
		         dcc.Loading(html.Div(id='ac-undershoot-limit', style={'color': 'blue'})), ], lg=2),
		dbc.Col([html.Div(children=html.Strong('Corridor width, deg: ')),
				 dcc.Loading(html.Div(id='ac-tcw', style={'color': 'blue'})), ], lg=2),
		dbc.Col([html.Div(children=html.Strong('Automatic Update:')),
				 dcc.RadioItems(id='ac-traj-auto-update', options = [
					 {'label': 'On', 'value': 'True'},
					 {'label': 'Off', 'value': 'False'},
				 ], value= 'True',
					inputStyle={"margin-right": "5px"},
					labelStyle={'display': 'inline-block', "margin-left": "10px"}),
			], lg=3),
	]),
	dbc.Row([
		dbc.Col(lg=2),
		dbc.Col(html.Div(id='ac-corridor-failed-message'), lg=8),
		dbc.Col(lg=2)
	]),
	dbc.Row([
			dbc.Col(lg=1),
			dbc.Col([
				dcc.Loading([
					dcc.Graph(id='ac-trajectory-figure')
				])
			], lg=10),
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
		html.H3('Click on one of the buttons above to start.'),
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
						html.H6("Pilot program of rapid mission architecture tool design for planetary science.")
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
	approach_layout,
	ac_layout,
	link_budget_layout,
	home_layout,
])

app.layout = main_layout

@app.callback(Output('approach-page-heading', 'children'),
			  Input('approach-page-location', 'pathname'))
def get_approach_page_title(pathname):
	if unquote(pathname[1:]) in 'approach':
		return html.H3(html.Strong(f'Approach Trajectory Calculator'))
	
	
@app.callback(Output('ac-page-heading', 'children'),
			  Input('ac-page-location', 'pathname'))
def get_ac_page_title(pathname):
	if unquote(pathname[1:]) in 'aerocapture':
		return html.H3(html.Strong(f'Aerocapture Trajectory Calculator'))


@app.callback(Output('approach-main-viz', 'children'),
			  Output('selected-periapsis', 'children'),
			  Output('selected-clock-angle', 'children'),
			  Output('selected-apoapsis', 'children'),
			  Output('orbit-insertion-dv', 'children'),
			  Output('approach-results-table', 'children'),
			  Output('entry-trajectory-figure', 'figure'),
			  Output('selected-mass', 'children'),
			  Output('selected-diameter', 'children'),
			  Output('selected-nose-radius', 'children'),
			  Output('ballistic-coefficient', 'children'),
			  Input('approach-planet-dropdown', 'value'),
			  Input('approach-vinf-input', 'value'),
			  Input('approach-is-entrysystem', 'value'),
			  Input('approach-periapsis-slider', 'value'),
			  Input('approach-psi-slider', 'value'),
			  Input('orbit-apoapsis-slider', 'value'),
			  Input('mass-slider', 'value'),
			  Input('diameter-slider', 'value'),
			  Input('nose-radius-slider', 'value'),
			  Input('approach-page-location', 'pathname'))
def get_approach_page_main_chart(planet, vinf_ICRF_kms, approach_type, periapsis_alt, psi_deg, apoapsis,
                                 mass, diameter, rn, pathname):
	if unquote(pathname[1:]) in 'approach':
		vinf_kms = np.array([float(i) for i in vinf_ICRF_kms.split(',')])

		periapsis_alt_km = float(periapsis_alt)
		psi_rad = float(psi_deg)*np.pi/180
		
		mass = float(mass)
		diam = float(diameter)
		area = 0.25*np.pi*diam**2
		CD = 1.0
		rn = float(rn)
		beta = mass / (area*CD)
		planet1 = Planet(planet.upper())
		if approach_type == 'probe':
			probe1 = Approach(planet.upper(),
							  v_inf_vec_icrf_kms=vinf_kms,
							  rp=(planet1.RP/1000 + periapsis_alt_km) * 1e3, psi=psi_rad,
							  is_entrySystem=True, h_EI=1000e3)

			theta_star_arr_probe1 = np.linspace(-1.8, probe1.theta_star_entry, 1001)
			pos_vec_bi_arr_probe1 = probe1.pos_vec_bi(theta_star_arr_probe1) / planet1.RP

			x_arr_probe1 = pos_vec_bi_arr_probe1[0][:]
			y_arr_probe1 = pos_vec_bi_arr_probe1[1][:]
			z_arr_probe1 = pos_vec_bi_arr_probe1[2][:]
			sphereSource = vtkSphereSource()
			sphereSource.SetCenter(0.0, 0.0, 0.0)
			sphereSource.SetRadius(1.0)
			# Make the surface smooth.
			sphereSource.SetPhiResolution(100)
			sphereSource.SetThetaResolution(100)
			sphereSource.Update()


			dataset = sphereSource.GetOutput()

			u = np.linspace(0, 2 * np.pi, 500)
			v = np.linspace(0, np.pi, 500)
			x_ring_1 = 1.5 * np.cos(u)
			y_ring_1 = 1.5 * np.sin(u)
			z_ring_1 = 0.0 * np.cos(u)

			x_ring_2 = 1.2 * np.cos(u)
			y_ring_2 = 1.2 * np.sin(u)
			z_ring_2 = 0.0 * np.cos(u)

			points1 = []
			points2 = []
			points3 = []
			for i in range(500):
				points1.append([x_ring_1[i], y_ring_1[i], z_ring_1[i]])
				points2.append([x_ring_2[i], y_ring_2[i], z_ring_2[i]])

			for j in range(0, len(x_arr_probe1)):
				points3.append([x_arr_probe1[j], y_arr_probe1[j], z_arr_probe1[j]])


			points_flat1 = [item for sublist in points1 for item in sublist]
			points_flat2 = [item for sublist in points2 for item in sublist]
			points_flat3 = [item for sublist in points3 for item in sublist]


			# Use helper to get a mesh structure that can be passed as-is to a Mesh
			# RTData is the name of the field
			mesh_state = to_mesh_state(dataset)

			content = dash_vtk.View([
				dash_vtk.GeometryRepresentation([
					dash_vtk.Mesh(state=mesh_state)
				], property={'color': planet_color_map[planet]}),
				dash_vtk.PointCloudRepresentation(xyz=points_flat1, property={'color': (1, 1, 1)}),
				dash_vtk.PointCloudRepresentation(xyz=points_flat2, property={'color': (1, 1, 1)}),
				dash_vtk.PointCloudRepresentation(xyz=points_flat3, property={'color': (1, 0, 0)})
			], background=[0, 0, 0])

			OI_DV = None
			
			entry_state = {
				'Altitude, km': round(probe1.h_EI/1000),
				'Longitude, deg (BI)': round(probe1.longitude_entry_bi*180/np.pi, 2),
				'Latitude, deg (BI)': round(probe1.latitude_entry_bi*180/np.pi, 2),
				'Speed, km/s (atm. rel)': round(probe1.v_entry_atm_mag/1000, 2),
				'EFPA, deg. (atm. rel)': round(probe1.gamma_entry_atm*180/np.pi, 2),
				'Heading angle, deg.': round(probe1.heading_entry_atm*180/np.pi, 2)
			}
			
			entry_state_df = pd.DataFrame([entry_state])
			
			planet1.h_skip = 1000.0E3
			planet1.h_trap = -100.0E3
			planet1.loadAtmosphereModel(
				planet_atmdata_map[planet],
				planet_h_col_map[planet],
				planet_t_col_map[planet],
				planet_p_col_map[planet],
				planet_rho_col_map[planet],
				heightInKmFlag=planet_h_km_flag[planet])
			
			vehicle1 = Vehicle(
				'probe', mass, beta, 0.0, area, 0.0, rn, planet1)
			vehicle1.setInitialState(
				entry_state['Altitude, km'],
				entry_state['Longitude, deg (BI)'],
				entry_state['Latitude, deg (BI)'],
				entry_state['Speed, km/s (atm. rel)'],
				entry_state['Heading angle, deg.'],
				entry_state['EFPA, deg. (atm. rel)'], 0.0, 0.0)
			
			vehicle1.setSolverParams(1E-6)
			vehicle1.propogateEntry(30 * 60.0, 0.1, 0.0)
			
			table = DataTable(columns=[{'name': col, 'id': col} for col in entry_state_df.columns],
			                  data=entry_state_df.to_dict('records'),
			                  style_header={'whiteSpace': 'normal'},
			                  fixed_rows={'headers': True},
			                  virtualization=True,
			                  style_table={'height': '65px', 'width': '1430px'},
			                  sort_action='none',
			                  filter_action='none',
			                  export_format='none',
			                  style_cell={'minWidth': '180px'})
			
			fig = make_subplots(
				rows=2, cols=2,
				subplot_titles=("Altitude, km", "Deceleration, g", "Heat rate, W/cm2", "Heat load, kJ/cm2"),
				vertical_spacing=0.10
			)
			
			fig.add_trace(go.Scatter(x=vehicle1.t_minc, y=vehicle1.h_kmc, showlegend=False),
			              row=1, col=1)
			
			fig.add_trace(go.Scatter(x=vehicle1.t_minc, y=vehicle1.acc_net_g, showlegend=False),
			              row=1, col=2)
			
			fig.add_trace(go.Scatter(x=vehicle1.t_minc, y=vehicle1.q_stag_total, showlegend=False),
			              row=2, col=1)
			
			fig.add_trace(go.Scatter(x=vehicle1.t_minc, y=vehicle1.heatload/1000, showlegend=False),
			              row=2, col=2)
			
			fig.update_layout(height=750,
			                  title_text="Entry Trajectory vs Time (mins)")

		elif approach_type == 'orbiter':

			probe1 = Approach(planet.upper(),
							  v_inf_vec_icrf_kms=vinf_kms,
							  rp=(planet1.RP / 1000 + periapsis_alt_km) * 1e3, psi=psi_rad,
							  is_entrySystem=False)

			theta_star_arr_probe1 = np.linspace(-1.8, 0, 1001)
			pos_vec_bi_arr_probe1 = probe1.pos_vec_bi(theta_star_arr_probe1) / planet1.RP

			x_arr_probe1 = pos_vec_bi_arr_probe1[0][:]
			y_arr_probe1 = pos_vec_bi_arr_probe1[1][:]
			z_arr_probe1 = pos_vec_bi_arr_probe1[2][:]
			sphereSource = vtkSphereSource()
			sphereSource.SetCenter(0.0, 0.0, 0.0)
			sphereSource.SetRadius(1.0)
			# Make the surface smooth.
			sphereSource.SetPhiResolution(100)
			sphereSource.SetThetaResolution(100)
			sphereSource.Update()

			dataset = sphereSource.GetOutput()

			u = np.linspace(0, 2 * np.pi, 500)
			v = np.linspace(0, np.pi, 500)
			x_ring_1 = 1.5 * np.cos(u)
			y_ring_1 = 1.5 * np.sin(u)
			z_ring_1 = 0.0 * np.cos(u)

			x_ring_2 = 1.2 * np.cos(u)
			y_ring_2 = 1.2 * np.sin(u)
			z_ring_2 = 0.0 * np.cos(u)

			points1 = []
			points2 = []
			points3 = []
			for i in range(500):
				points1.append([x_ring_1[i], y_ring_1[i], z_ring_1[i]])
				points2.append([x_ring_2[i], y_ring_2[i], z_ring_2[i]])

			for j in range(0, len(x_arr_probe1)):
				points3.append([x_arr_probe1[j], y_arr_probe1[j], z_arr_probe1[j]])

			points_flat1 = [item for sublist in points1 for item in sublist]
			points_flat2 = [item for sublist in points2 for item in sublist]
			points_flat3 = [item for sublist in points3 for item in sublist]

			# Use helper to get a mesh structure that can be passed as-is to a Mesh
			# RTData is the name of the field
			mesh_state = to_mesh_state(dataset)

			orbiter = PropulsiveOrbiter(approach=probe1, apoapsis_alt_km=float(apoapsis))
			orbiter.compute_orbit_trajectory(num_points=2000)

			points4 = []
			for j in range(0, len(orbiter.x_orbit_arr)):
				points4.append([orbiter.x_orbit_arr[j], orbiter.y_orbit_arr[j], orbiter.z_orbit_arr[j]])

			points_flat4 = [item for sublist in points4 for item in sublist]

			content = dash_vtk.View([
				dash_vtk.GeometryRepresentation([
					dash_vtk.Mesh(state=mesh_state)
				], property={'color': planet_color_map[planet]}),
				dash_vtk.PointCloudRepresentation(xyz=points_flat1, property={'color': (1, 1, 1)}),
				dash_vtk.PointCloudRepresentation(xyz=points_flat2, property={'color': (1, 1, 1)}),
				dash_vtk.PointCloudRepresentation(xyz=points_flat3, property={'color': (1, 0, 0)}),
				dash_vtk.PointCloudRepresentation(xyz=points_flat4, property={'color': (1, 0, 1)})
			], background=[0, 0, 0])

			if not math.isnan(orbiter.DV_OI_mag):
				OI_DV = round(orbiter.DV_OI_mag)
			else:
				OI_DV = None
			orbital_elements = {
				'Perigee, km': round((orbiter.a*(1 - orbiter.e))/1000 - orbiter.approach.planetObj.RP/1000),
				'Apogee, km': round((orbiter.a*(1 + orbiter.e))/1000 - orbiter.approach.planetObj.RP/1000),
				'Inclination, deg': round(orbiter.i*180/np.pi, 2),
				'RAAN, deg': round(orbiter.OMEGA*180/np.pi, 2),
				'AoP, deg': round(orbiter.omega*180/np.pi, 2),
			}
			
			orbital_elements_df = pd.DataFrame([orbital_elements])
			
			table = DataTable(columns=[{'name': col, 'id': col} for col in orbital_elements_df.columns],
			                  data=orbital_elements_df.to_dict('records'),
			                  style_header={'whiteSpace': 'normal'},
			                  fixed_rows={'headers': True},
			                  virtualization=True,
			                  style_table={'height': '65px', 'width': '1430px'},
			                  sort_action='none',
			                  filter_action='none',
			                  export_format='none',
			                  style_cell={'minWidth': '150px'})
			
			fig = make_empty_fig()
			
		elif approach_type == 'flyby':
			
			probe1 = Approach(planet.upper(),
			                  v_inf_vec_icrf_kms=vinf_kms,
			                  rp=(planet1.RP / 1000 + periapsis_alt_km) * 1e3, psi=psi_rad,
			                  is_entrySystem=False, h_EI=1000e3)
			
			theta_star_arr_probe1 = np.linspace(-1.8, 1.8, 1001)
			pos_vec_bi_arr_probe1 = probe1.pos_vec_bi(theta_star_arr_probe1) / planet1.RP
			
			x_arr_probe1 = pos_vec_bi_arr_probe1[0][:]
			y_arr_probe1 = pos_vec_bi_arr_probe1[1][:]
			z_arr_probe1 = pos_vec_bi_arr_probe1[2][:]
			sphereSource = vtkSphereSource()
			sphereSource.SetCenter(0.0, 0.0, 0.0)
			sphereSource.SetRadius(1.0)
			# Make the surface smooth.
			sphereSource.SetPhiResolution(100)
			sphereSource.SetThetaResolution(100)
			sphereSource.Update()
			
			dataset = sphereSource.GetOutput()
			
			u = np.linspace(0, 2 * np.pi, 500)
			v = np.linspace(0, np.pi, 500)
			x_ring_1 = 1.5 * np.cos(u)
			y_ring_1 = 1.5 * np.sin(u)
			z_ring_1 = 0.0 * np.cos(u)
			
			x_ring_2 = 1.2 * np.cos(u)
			y_ring_2 = 1.2 * np.sin(u)
			z_ring_2 = 0.0 * np.cos(u)
			
			points1 = []
			points2 = []
			points3 = []
			for i in range(500):
				points1.append([x_ring_1[i], y_ring_1[i], z_ring_1[i]])
				points2.append([x_ring_2[i], y_ring_2[i], z_ring_2[i]])
			
			for j in range(0, len(x_arr_probe1)):
				points3.append([x_arr_probe1[j], y_arr_probe1[j], z_arr_probe1[j]])
			
			points_flat1 = [item for sublist in points1 for item in sublist]
			points_flat2 = [item for sublist in points2 for item in sublist]
			points_flat3 = [item for sublist in points3 for item in sublist]
			
			# Use helper to get a mesh structure that can be passed as-is to a Mesh
			# RTData is the name of the field
			mesh_state = to_mesh_state(dataset)
			
			content = dash_vtk.View([
				dash_vtk.GeometryRepresentation([
					dash_vtk.Mesh(state=mesh_state)
				], property={'color': planet_color_map[planet]}),
				dash_vtk.PointCloudRepresentation(xyz=points_flat1, property={'color': (1, 1, 1)}),
				dash_vtk.PointCloudRepresentation(xyz=points_flat2, property={'color': (1, 1, 1)}),
				dash_vtk.PointCloudRepresentation(xyz=points_flat3, property={'color': (1, 0, 0)})
			], background=[0, 0, 0])
			
			OI_DV = None
			table = None
			
			fig = make_empty_fig()
		
		return ([content], round(periapsis_alt_km), round(psi_deg), round(apoapsis), html.H4(html.Strong(OI_DV)),
		        table, fig, round(mass), round(diam, 1), round(rn, 1), round(beta,1))


@app.callback(
	Output('ac-selected-apoapsis', 'children'),
	Output('ac-selected-mass', 'children'),
	Output('ac-selected-diameter', 'children'),
	Output('ac-selected-nose-radius', 'children'),
	Output('ac-selected-LD', 'children'),
	Output('ac-selected-BCR', 'children'),
	Output('ac-ballistic-coefficient', 'children'),
	Input('ac-planet-dropdown', 'value'),
	Input('ac-vehicle-type', 'value'),
	Input('ac-apoapsis-slider', 'value'),
	Input('ac-mass-slider', 'value'),
	Input('ac-diameter-slider', 'value'),
	Input('ac-nose-radius-slider', 'value'),
	Input('ac-LD-slider', 'value'),
	Input('ac-bcr-slider', 'value'),
	Input('ac-page-location', 'pathname'))
def update_ac_vehicle_design(planet, type, apoapsis, mass, diam, rn, LD, bcr, pathname):
	if unquote(pathname[1:]) in 'aerocapture':
		mass = float(mass)
		diam = float(diam)
		area = 0.25 * np.pi * diam ** 2
		CD = 1.0
		rn = float(rn)
		LD = float(LD)
		beta = mass / (area * CD)
		bcr = float(bcr)
		
		return round(apoapsis), round(mass), round(diam, 1), round(rn, 1), round(LD, 2), round(bcr, 1), round(beta, 1)


@app.callback(
	Output('ac-overshoot-limit', 'children'),
	Output('ac-undershoot-limit', 'children'),
	Output('ac-tcw', 'children'),
	Output('ac-corridor-failed-message', 'children'),
	Output('ac-trajectory-figure', 'figure'),
	Input('ac-planet-dropdown', 'value'),
	Input('ac-vehicle-type', 'value'),
	Input('ac-apoapsis-slider', 'value'),
	Input('ac-mass-slider', 'value'),
	Input('ac-diameter-slider', 'value'),
	Input('ac-nose-radius-slider', 'value'),
	Input('ac-LD-slider', 'value'),
	Input('ac-bcr-slider', 'value'),
	Input('ac-entry-altitude', 'value'),
	Input('ac-entry-longitude', 'value'),
	Input('ac-entry-latitude', 'value'),
	Input('ac-entry-speed', 'value'),
	Input('ac-entry-heading', 'value'),
	Input('ac-page-location', 'pathname'),
	Input('ac-traj-auto-update', 'value'))
def update_ac_charts(planet, type, apoapsis, mass, diam, rn, LD, bcr,
                             alt, theta, phi, v, psi, pathname, auto_update):
	if auto_update == 'False':
		raise PreventUpdate
	if unquote(pathname[1:]) in 'aerocapture':
		apoapsis = float(apoapsis)
		mass = float(mass)
		diam = float(diam)
		area = 0.25 * np.pi * diam ** 2
		CD = 1.0
		rn = float(rn)
		LD = float(LD)
		beta = mass / (area * CD)
		bcr = float(bcr)
		alt = float(alt)
		theta = float(theta)
		phi = float(phi)
		v = float(v)
		psi = float(psi)

		
		planet1 = Planet(planet.upper())
		planet1.h_skip = alt*1e3
		planet1.loadAtmosphereModel(
			planet_atmdata_map[planet],
			planet_h_col_map[planet],
			planet_t_col_map[planet],
			planet_p_col_map[planet],
			planet_rho_col_map[planet],
			heightInKmFlag=planet_h_km_flag[planet]
		)
		if type == 'lift-modulation':
			vehicle = Vehicle('Apollo', mass, beta, LD, area, 0.0, rn, planet1)
			vehicle.setInitialState(alt, theta, phi, v, psi, -4.5, 0.0, 0.0)
			vehicle.setSolverParams(1E-6)
			
			overShootLimit, exitflag_os = vehicle.findOverShootLimit(2400.0, 0.1, -30.0, -4.0, 1E-5, apoapsis)
			underShootLimit, exitflag_us = vehicle.findUnderShootLimit(2400.0, 0.1, -30.0, -4.0, 1E-5, apoapsis)
			tcw = overShootLimit - underShootLimit
			
			if exitflag_us == 1 and exitflag_os == 1:
				status = False
				
				# propogate the overshoot and undershoot trajectories
				vehicle.setInitialState(alt, theta, phi, v, psi, overShootLimit, 0.0, 0.0)
				vehicle.propogateEntry(2400.0, 0.1, 180.0)
				
				# Extract and save variables to plot
				t_min_os = vehicle.t_minc
				h_km_os = vehicle.h_kmc
				acc_net_g_os = vehicle.acc_net_g
				q_stag_os = vehicle.q_stag_total
				heatload_os = vehicle.heatload / 1000
				
				vehicle.setInitialState(alt, theta, phi, v, psi, underShootLimit, 0.0, 0.0)
				vehicle.propogateEntry(2400.0, 0.1, 0.0)
				
				# Extract and save variable to plot
				t_min_us = vehicle.t_minc
				h_km_us = vehicle.h_kmc
				acc_net_g_us = vehicle.acc_net_g
				q_stag_us = vehicle.q_stag_total
				heatload_us = vehicle.heatload /1000
				
				# propagate the nominal trajectory with L/D = 0
				vehicle.LD = 0
				vehicle.setInitialState(alt, theta, phi, v, psi, 0.5*(overShootLimit + underShootLimit), 0.0, 0.0)
				vehicle.propogateEntry(2400.0, 0.1, 0.0)
				
				# Extract and save variable to plot
				t_min_no = vehicle.t_minc
				h_km_no = vehicle.h_kmc
				acc_net_g_no = vehicle.acc_net_g
				q_stag_no = vehicle.q_stag_total
				heatload_no = vehicle.heatload / 1000
				
				fig = make_subplots(
					rows=2, cols=2,
					subplot_titles=("Altitude, km", "Deceleration, g", "Heat rate, W/cm2", "Heat load, kJ/cm2"),
					vertical_spacing=0.10,
					shared_xaxes=True
				)
				
				fig.add_trace(go.Scatter(x=t_min_os, y=h_km_os, line=dict(color='blue'), showlegend=False),
				              row=1, col=1)
				
				fig.add_trace(go.Scatter(x=t_min_us, y=h_km_us, line=dict(color='red'), showlegend=False),
				              row=1, col=1)
				
				fig.add_trace(go.Scatter(x=t_min_no, y=h_km_no, line=dict(color='green'),  showlegend=False),
				              row=1, col=1)
				
				fig.add_trace(go.Scatter(x=t_min_os, y=acc_net_g_os, line=dict(color='blue'), showlegend=False),
				              row=1, col=2)
				
				fig.add_trace(go.Scatter(x=t_min_us, y=acc_net_g_us, line=dict(color='red'), showlegend=False),
				              row=1, col=2)
				
				fig.add_trace(go.Scatter(x=t_min_no, y=acc_net_g_no, line=dict(color='green'), showlegend=False),
				              row=1, col=2)
				
				fig.add_trace(go.Scatter(x=t_min_os, y=q_stag_os, line=dict(color='blue'), showlegend=False),
				              row=2, col=1)
				
				fig.add_trace(go.Scatter(x=t_min_us, y=q_stag_us, line=dict(color='red'), showlegend=False),
				              row=2, col=1)
				
				fig.add_trace(go.Scatter(x=t_min_no, y=q_stag_no, line=dict(color='green'), showlegend=False),
				              row=2, col=1)
				
				fig.add_trace(go.Scatter(x=t_min_os, y=heatload_os, line=dict(color='blue'), showlegend=False),
				              row=2, col=2)
				
				fig.add_trace(go.Scatter(x=t_min_us, y=heatload_us, line=dict(color='red'), showlegend=False),
				              row=2, col=2)
				
				fig.add_trace(go.Scatter(x=t_min_no, y=heatload_no, line=dict(color='green'), showlegend=False),
				              row=2, col=2)
				
				fig.update_layout(height=750,
				                  title_text="Undershoot [red], nominal [green] and overshoot [blue] trajectories vs time (min)")
			
			else:
				status = True
				overShootLimit = 0
				underShootLimit = 0
				fig = make_empty_fig()
				
		elif type == 'drag-modulation':
			vehicle = Vehicle('Apollo', mass, beta, 0.0, area, 0.0, rn, planet1)
			vehicle.setInitialState(alt, theta, phi, v, psi, -4.5, 0.0, 0.0)
			vehicle.setSolverParams(1E-6)
			vehicle.setDragModulationVehicleParams(beta, bcr)
			
			overShootLimit, exitflag_os = vehicle.findOverShootLimitD(2400.0, 0.1, -20.0, -4.0, 1E-5, apoapsis)
			underShootLimit, exitflag_us = vehicle.findUnderShootLimitD(2400.0, 0.1, -20.0, -4.0, 1E-5, apoapsis)
			tcw = overShootLimit - underShootLimit
			
			if exitflag_us == 1 and exitflag_os == 1:
				status = False
				
				# propogate the overshoot and undershoot trajectories
				vehicle = Vehicle('Apollo', mass, beta, 0.0, area, 0.0, rn, planet1)
				vehicle.setInitialState(alt, theta, phi, v, psi, overShootLimit, 0.0, 0.0)
				vehicle.setSolverParams(1E-6)
				vehicle.propogateEntry(2400.0, 0.1, 0.0)
				
				# Extract and save variables to plot
				t_min_os = vehicle.t_minc
				h_km_os = vehicle.h_kmc
				acc_net_g_os = vehicle.acc_net_g
				q_stag_os = vehicle.q_stag_total
				heatload_os = vehicle.heatload / 1000
				
				vehicle = Vehicle('Apollo', mass, beta*bcr, 0.0, area, 0.0, rn, planet1)
				vehicle.setInitialState(alt, theta, phi, v, psi, underShootLimit, 0.0, 0.0)
				vehicle.setSolverParams(1E-6)
				vehicle.propogateEntry(2400.0, 0.1, 0.0)
				
				# Extract and save variable to plot
				t_min_us = vehicle.t_minc
				h_km_us = vehicle.h_kmc
				acc_net_g_us = vehicle.acc_net_g
				q_stag_us = vehicle.q_stag_total
				heatload_us = vehicle.heatload / 1000
				

				
				fig = make_subplots(
					rows=2, cols=2,
					subplot_titles=("Altitude, km", "Deceleration, g", "Heat rate, W/cm2", "Heat load, kJ/cm2"),
					vertical_spacing=0.10,
					shared_xaxes=True
				)
				
				fig.add_trace(go.Scatter(x=t_min_os, y=h_km_os, line=dict(color='blue'), showlegend=False),
				              row=1, col=1)
				
				fig.add_trace(go.Scatter(x=t_min_us, y=h_km_us, line=dict(color='red'), showlegend=False),
				              row=1, col=1)
				
				
				fig.add_trace(go.Scatter(x=t_min_os, y=acc_net_g_os, line=dict(color='blue'), showlegend=False),
				              row=1, col=2)
				
				fig.add_trace(go.Scatter(x=t_min_us, y=acc_net_g_us, line=dict(color='red'), showlegend=False),
				              row=1, col=2)
				
				fig.add_trace(go.Scatter(x=t_min_os, y=q_stag_os, line=dict(color='green'), showlegend=False),
				              row=2, col=1)
				
				
				fig.add_trace(go.Scatter(x=t_min_os, y=heatload_os, line=dict(color='green'), showlegend=False),
				              row=2, col=2)
				
				
				fig.update_layout(height=750,
				                  title_text="Undershoot [red], nominal [green] and overshoot [blue] trajectories vs time (min)")
				
			else:
				status = True
				overShootLimit = 0
				underShootLimit = 0
				fig = make_empty_fig()
				

		return round(overShootLimit, 3), round(underShootLimit, 3), \
		       round(tcw, 2), failed_ac_corridor(status), fig
		
	

@app.callback(Output('main_content', 'children'),
			  Input('main_location', 'pathname'))
def display_content(pathname):
	if unquote(pathname[1:]) in planets:
		return planet_layout
	if unquote(pathname[1:]) == 'launch_perf':
		return launcher_layout
	if unquote(pathname[1:]) == 'approach':
		return approach_layout
	if unquote(pathname[1:]) == 'aerocapture':
		return ac_layout
	if unquote(pathname[1:]) == 'link_budget':
		return link_budget_layout
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
	
	
@app.callback(Output('link-budget-heading', 'children'),
			  Input('link_budget_location', 'pathname'))
def get_link_budget_page_title(pathname):
	if unquote(pathname[1:]) in 'link_budget':
		return html.H3(html.Strong(f'Link Budget Calculator'))
	




@app.callback(Output('launch-mass-vs-c3-chart', 'figure'),
			  Input('launch_page_location', 'pathname'),
			  Input('launch-vehicles-checkbox', 'value'), )
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





@app.callback(Output('approach-planet-dropdown', 'options'),
			  Output('approach-planet-dropdown', 'value'),
			  Input('approach-page-location', 'pathname'))
def update_approach_planet_dropdown(pathname):
	planet_options = [{'label': planet, 'value': planet} for planet in planets]
	planet_value = planet_options[0]['value']
	return planet_options, planet_value


@app.callback(Output('ac-planet-dropdown', 'options'),
			  Output('ac-planet-dropdown', 'value'),
			  Input('ac-page-location', 'pathname'))
def update_ac_planet_dropdown(pathname):
	planet_options = [{'label': planet, 'value': planet} for planet in planets_ac]
	planet_value = 'Uranus'
	return planet_options, planet_value

@app.callback(Output('ac-entry-altitude', 'value'),
			  Input('ac-planet-dropdown', 'value'))
def update_ac_entry_altitude(planet):
	value = planet_entry_int_alt_km[planet]
	return value

@app.callback(Output('ac-entry-speed', 'value'),
			  Input('ac-planet-dropdown', 'value'))
def update_ac_entry_altitude(planet):
	value = planet_entry_speed_kms[planet]
	return value

@app.callback(Output('ac-apoapsis-slider', 'value'),
			  Input('ac-planet-dropdown', 'value'))
def update_ac_apoapsis(planet):
	value = planet_target_apoapsis[planet]
	return value


@app.callback(Output('link-budget-frequency', 'value'),
              Output('link-budget-pt', 'value'),
              Output('link-budget-gt', 'value'),
			  Output('link-budget-gr', 'value'),
			  Output('link-budget-ts', 'value'),
			  Output('link-budget-range', 'value'),
			  Output('link-budget-lt', 'value'),
			  Output('link-budget-lsc', 'value'),
			  Output('link-budget-la', 'value'),
			  Output('link-budget-lother', 'value'),
			  Output('link-budget-ldsn', 'value'),
			  Output('link-budget-rate-kbps', 'value'),
			  Output('link-budget-eb-n0-required', 'value'),
              Input('link-budget-presets', 'value'))
def update_link_budget_values(id):
	x = link_budget_presets[id]
	return x['freq'], x['pt'], x['gt'], x['gr'], x['ts'], x['range'], x['lt'], x['lsc'], \
	       x['la'], x['lother'], x['ldsn'], x['rate'], x['eb-n0-required']
	


@app.callback(Output('link-budget-eb-n0-computed', 'children'),
			  Output('link-budget-eb-n0-margin', 'children'),
			  Input('link-budget-auto-update', 'value'),
			  Input('link-budget-frequency', 'value'),
              Input('link-budget-pt', 'value'),
              Input('link-budget-gt', 'value'),
			  Input('link-budget-gr', 'value'),
			  Input('link-budget-ts', 'value'),
			  Input('link-budget-range', 'value'),
			  Input('link-budget-lt', 'value'),
			  Input('link-budget-lsc', 'value'),
			  Input('link-budget-la', 'value'),
			  Input('link-budget-lother', 'value'),
			  Input('link-budget-ldsn', 'value'),
			  Input('link-budget-rate-kbps', 'value'),
			  Input('link-budget-eb-n0-required', 'value'))
def compute_link_budget(auto_update, freq, pt, gt, gr, ts, range, lt, lsc, la, l_other, l_dsn, rate, eb_n0_req):
	if auto_update == 'False':
		raise PreventUpdate
	link = Link(freq=freq, Pt=pt, Gt_dBi=gt, Gr_dBi=gr, Ts=ts, range_km=range, Lt_dB=lt, L_sc_dB=lsc, La_dB=la,
	            L_other_dB=l_other, L_DSN_dB=l_dsn, rate_kbps=rate, Eb_N0_req=eb_n0_req)
	
	return html.H4(html.Strong(round(link.EB_N0, 2))), html.H4(html.Strong(round(link.EB_N0_margin, 2)))
	
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
	C3_options = [{'label': C3, 'value': C3} for C3 in np.arange(5, np.ceil(df.LC3.max()) + 1, step=5)]
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
	TOF_options = [{'label': TOF, 'value': TOF} for TOF in
				   np.arange(np.ceil(df.TOF.min()), np.ceil(df.TOF.max()) + 1, step=1)]
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
	Avinf_options = [{'label': Avinf, 'value': Avinf} for Avinf in
					 np.arange(np.ceil(df.Avinf.min()), np.ceil(df.Avinf.max()) + 1, step=1)]
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
	                    hover_data=["Date", "LC3", "TOF", "DSMdv", "Avinf", "Type"],
						title="C3 vs. Launch Date",
						height=700)

	c3_fig.update_layout(xaxis_title="Launch Date",
						 yaxis_title="Launch C3",
						 font=dict(size=13))

	tof_fig = px.scatter(df_1, x='Date', y='TOF',
						 color='Gravity Assist Path',
						 hover_name='Gravity Assist Path',
						 hover_data=["Date", "LC3", "TOF", "DSMdv", "Avinf", "Type"],
						 title="TOF vs. Launch Date",
						 height=700)

	tof_fig.update_layout(xaxis_title="Launch Date",
						  yaxis_title="ToF, years",
						  font=dict(size=13))

	avinf_fig = px.scatter(df_1, x='Date', y='Avinf',
						   color='Gravity Assist Path',
						   hover_name='Gravity Assist Path',
						   hover_data=["Date", "LC3", "TOF", "DSMdv", "Avinf", "Type"],
						   title="Arrival Vinf vs. Launch Date",
						   height=700)

	avinf_fig.update_layout(xaxis_title="Launch Date",
							yaxis_title="Vinf, km/s",
							font=dict(size=13))

	c3_tof_fig = px.scatter(df_1, x='TOF', y='LC3',
							color='Gravity Assist Path',
							hover_name='Gravity Assist Path',
							hover_data=["Date", "LC3", "TOF", "DSMdv", "Avinf", "Type"],
							title="C3 vs. TOF",
							height=700)

	c3_tof_fig.update_layout(xaxis_title="TOF, years",
							 yaxis_title="Launch C3",
							 font=dict(size=13))

	avinf_tof_fig = px.scatter(df_1, x='TOF', y='Avinf',
							   color='Gravity Assist Path',
							   hover_name='Gravity Assist Path',
							   hover_data=["Date", "LC3", "TOF", "DSMdv", "Avinf", "Type"],
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
