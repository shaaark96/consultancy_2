from dash import Dash, dash_table, dcc, html, Input, Output, State, callback_context
from dash_iconify import DashIconify
from PIL import Image
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import glob
import plotly.graph_objects as go

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, '/assets/custom.css'])

"""
-----------------------------------------------------------------------------------------
Section 1:
Data Import and Preparation
"""
# Data reading:
data_path = 'assets/all_fixation_data_cleaned_up.csv'
df = pd.read_csv(data_path, sep=';')

# Add "Task Duration in sec" (per User and Stimulus) to df:
task_duration = df.groupby(['user', 'CityMap', 'description'])['FixationDuration'].sum().reset_index()
task_duration['FixationDuration'] = task_duration['FixationDuration'] / 1000
df = pd.merge(df, task_duration, on=['user', 'CityMap', 'description'], suffixes=('', '_aggregated'))

# Add "Average Fixation Duration in sec" (per User and Stimulus) to df:
avg_fix_duration = df.groupby(['user', 'CityMap', 'description'])['FixationDuration'].mean().reset_index()
avg_fix_duration['FixationDuration'] = avg_fix_duration['FixationDuration'] / 1000
df = pd.merge(df, avg_fix_duration, on=['user', 'CityMap', 'description'], suffixes=('', '_avg'))

# Add Category for Task Duration:
df['TaskDurationCategory'] = pd.cut(df['FixationDuration_aggregated'], bins=[0, 10, float('inf')],
                                    labels=['<10 sec.', '>=10 sec.'])
#print('task_duration:')
#print(df['FixationDuration_aggregated'])
#print(df['FixationDuration_aggregated'].min)
#print(df['FixationDuration_aggregated'].max)
#print(df)

"""
-----------------------------------------------------------------------------------------
Section 2:
Definition of Dash Layout
"""
app.layout = html.Div([
    # Header and Theme-Mode:
    html.Div([
        html.Div([
            html.H1('Analysis of Eye-Tracking-Data'),
            html.H4('created on behalf of FHGR Chur, last updated: 20.06.2024')],
            className='header'),
        dcc.Dropdown(
            id='theme_dropdown',
            options=[
                {'label': 'Light Mode', 'value': 'light'},
                {'label': 'Dark Mode', 'value': 'dark'}],
            value='light',
            clearable=False,
            className='theme_dropdown',
        ),
        dcc.Store(id='current_theme', data='light'),
    ], className='first_container'),

    html.Div([
        # Start first column (Input, KPI, Histogram):
        html.Div([
            # Input-Containers:
            html.Div([
                # Visualization Type Buttons:
                html.Div([
                    html.H3([
                        DashIconify(icon="ion:bar-chart", width=16, height=16, style={"margin-right": "12px"}),
                        'Choose a Type of Visualization']),
                    html.Div([
                        html.Button('Boxplot', id='default_viz', n_clicks=0, className='viz_button'),
                        html.Button('Heat Map', id='heat_map', n_clicks=0, className='viz_button'),
                        html.Button('Gazeplot', id='gaze_plot', n_clicks=0, className='viz_button'),
                        html.Button('Correlation', id='scatter_plot', n_clicks=0, className='viz_button'),
                    ], id='button_viz_type', className='button_viz_type'),
                    dcc.Store(id='active-button', data='default_viz'),
                    html.Div(id='output-section'),
                ], className='third_container'),
            ], className='input_container'),

                # City Dropdown:
                html.Div([
                    html.H3([
                        DashIconify(icon="vaadin:train", width=16, height=16, style={"margin-right": "12px"}),
                        'Choose a City to explore in detail']),
                    dcc.Dropdown(
                        id='city_dropdown',
                        options=[{'label': city, 'value': city} for city in sorted(df['CityMap'].unique())],
                        placeholder='Select a City Map...',
                        value=None,
                        clearable=True,
                        className='dropdown'),
                ], className='second_container'),

            # Output-Container KPI-Table:
            html.Div([
                html.H3([
                    DashIconify(icon="fluent:arrow-trending-lines-24-filled", width=16, height=16,
                                style={"margin-right": "12px"}),
                    'Statistical Key Performance Indicators']),
                html.Div(id='table_container')
            ], className='fourth_container'),

            # Output-Container Histogram:
            html.Div([
                dcc.Graph(id='hist_taskduration'), #style={"height": "150px"}),
            ], className='seventh_container'),
        ], className='first_column'),

        # Start second column (Color Map):
        html.Div([
            # Output-Container Color Plot:
            html.Div([
                html.Img(
                    id='city_image_color'),
                dcc.Graph(id='gaze_plot_color'),
                dcc.Graph(id='heat_map_color'),
                dcc.Dropdown(id='dropdown_user_color', multi=True),
                dcc.RangeSlider(id='range_slider_color', min=1, max=50, step=1, value=[1, 50],
                                marks={i: f'{i}' for i in range(0, 51, 5)}),
                dcc.Graph(id='box_task_duration'),
                dcc.Graph(id='scatter_correlation_color')
            ], id='color_plot_area', className='fifth_container'),
        ], className='second_column'),

        # Start third column (Grey Map):
        html.Div([
            # Output-Container Grey Plot:
            html.Div([
                html.Img(
                    id='city_image_grey'),
                dcc.Graph(id='gaze_plot_grey'),
                dcc.Graph(id='heat_map_grey'),
                dcc.Dropdown(id='dropdown_user_grey', multi=True),
                dcc.RangeSlider(id='range_slider_grey', min=1, max=50, step=1, value=[1, 50],
                                marks = {i: f'{i}' for i in range(0, 51, 5)}),
                dcc.Graph(id='box_avg_fix_duration'),
                dcc.Graph(id='scatter_correlation_grey')
            ],  id='grey_plot_area', className='sixth_container'),
        ], className='third_column'),
    ], className='dash_container'),
], id='page_content', className='light_theme')

"""
-----------------------------------------------------------------------------------------
Section 3:
Definition of Interaction Elements
"""
# 3.1 - Define and keep active Viz-Button:
@app.callback(
    [Output('default_viz', 'className'),
     Output('heat_map', 'className'),
     Output('gaze_plot', 'className'),
     Output('scatter_plot', 'className'),
     Output('active-button', 'data')],
    [Input('default_viz', 'n_clicks'),
     Input('heat_map', 'n_clicks'),
     Input('gaze_plot', 'n_clicks'),
     Input('scatter_plot', 'n_clicks')],
    [State('active-button', 'data')]
)

def update_active_button(btn1, btn2, btn3, btn4, active_btn):
    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'default_viz'
    return [
        'viz_button active' if button_id == 'default_viz' else 'viz_button',
        'viz_button active' if button_id == 'heat_map' else 'viz_button',
        'viz_button active' if button_id == 'gaze_plot' else 'viz_button',
        'viz_button active' if button_id == 'scatter_plot' else 'viz_button',
        button_id
    ]

# 3.2 - Update Output Section and Plot Area based on active button, part I:
@app.callback(
    Output('output-section', 'children'),
    Input('active-button', 'data')
)
def update_output(active_button):
    if active_button == 'default_viz':
        return ''
    elif active_button == 'heat_map':
        return ''
    elif active_button == 'gaze_plot':
        return ''
    elif active_button == 'scatter_plot':
        return ''
    else:
        return ''

# 3.3 - Update Output Section and Plot Area based on active button, part II:
@app.callback(
    [Output('color_plot_area', 'children'),
     Output('grey_plot_area', 'children')],
    [Input('active-button', 'data'),
     Input('city_dropdown', 'value')]
)
def update_plot_area(visualization_type, selected_city):
    if visualization_type in ['gaze_plot', 'heat_map']:
        min_val_color, max_val_color, value_range_color, marks_color = update_range_slider_color(selected_city)
        min_val_grey, max_val_grey, value_range_grey, marks_grey = update_range_slider_grey(selected_city)

        return [
            dcc.Graph(id=f'{visualization_type}_color'),
            dcc.Dropdown(id='dropdown_user_color', value=None, multi=True, placeholder='filter by User(s)...'),
            html.P('filter by Task Duration:'),
            dcc.RangeSlider(id='range_slider_color',
                            min=min_val_color, max=max_val_color, value=value_range_color, marks=marks_color)
        ], [
            dcc.Graph(id=f'{visualization_type}_grey'),
            dcc.Dropdown(id='dropdown_user_grey', value=None, multi=True, placeholder='filter by User(s)...'),
            html.P('filter by Task Duration:'),
            dcc.RangeSlider(id='range_slider_grey',
                            min=min_val_grey, max=max_val_grey, value=value_range_grey, marks=marks_grey)
        ]
    elif visualization_type == 'scatter_plot':
        return [
            dcc.Graph(id='scatter_correlation_color')
        ], [
            dcc.Graph(id='scatter_correlation_grey')
        ]
    else:
        return [
            dcc.Graph(id='box_task_duration')
        ], [
            dcc.Graph(id='box_avg_fix_duration')
        ]

# 3.4 - Update Dropdown-Filters in plot area, based on selected city:
@app.callback(
    [Output('dropdown_user_color', 'options'),
     Output('dropdown_user_grey', 'options')],
    [Input('city_dropdown', 'value')]
)
def update_user_dropdowns(selected_city):
    if selected_city:
        # Filter users based on the selected city and description
        filtered_users_color = df[(df['CityMap'] == selected_city) & (df['description'] == 'color')]['user'].unique()
        filtered_users_grey = df[(df['CityMap'] == selected_city) & (df['description'] == 'grey')]['user'].unique()

        # Convert filtered users to dropdown options
        color_options = [{'label': user, 'value': user} for user in filtered_users_color]
        grey_options = [{'label': user, 'value': user} for user in filtered_users_grey]

        return color_options, grey_options

    return [[], []]

# 3.5 - Update Range-Slider in plot area, based on selected city and viz-type:
def to_int(value):
    return int(value) if not pd.isna(value) else 0

def update_range_slider(selected_city, description, buffer=5):
    if selected_city:
        filtered_df = df[(df['CityMap'] == selected_city) & (df['description'] == description)]
        if not filtered_df.empty:
            min_value = to_int(filtered_df['FixationDuration_aggregated'].min())
            max_value = to_int(filtered_df['FixationDuration_aggregated'].max())
            # Fester Wert von 2 zur Max-Grenze hinzufügen
            max_value_with_buffer = max_value + buffer
            marks = {i: f'{i}' for i in range(min_value, max_value_with_buffer + 1, max(1, (max_value_with_buffer - min_value) // 5))}
            return min_value, max_value_with_buffer, [min_value, max_value_with_buffer], marks
        else:
            return 0, 0, [0, 0], {0: '0'}
    else:
        global_min = to_int(df['FixationDuration_aggregated'].min())
        global_max = to_int(df['FixationDuration_aggregated'].max())
        # Fester Wert von 2 zur globalen Max-Grenze hinzufügen
        global_max_with_buffer = global_max + buffer
        marks = {i: f'{i}' for i in range(global_min, global_max_with_buffer + 1, max(1, (global_max_with_buffer - global_min) // 5))}
        return global_min, global_max_with_buffer, [global_min, global_max_with_buffer], marks

# 3.5.1 - Update color Slider:
def update_range_slider_color(selected_city):
    return update_range_slider(selected_city, 'color')

# 3.5.2 - Update grey Slider:
def update_range_slider_grey(selected_city):
    return update_range_slider(selected_city, 'grey')

# 3.5.3 - Callback for both Slider:
@app.callback(
    [Output('range_slider_color', 'min'),
     Output('range_slider_color', 'max'),
     Output('range_slider_color', 'value'),
     Output('range_slider_color', 'marks'),
     Output('range_slider_grey', 'min'),
     Output('range_slider_grey', 'max'),
     Output('range_slider_grey', 'value'),
     Output('range_slider_grey', 'marks')],
    [Input('city_dropdown', 'value')]
)
def update_range_sliders(selected_city):
    min_color, max_color, value_color, marks_color = update_range_slider_color(selected_city)
    min_grey, max_grey, value_grey, marks_grey = update_range_slider_grey(selected_city)
    return min_color, max_color, value_color, marks_color, min_grey, max_grey, value_grey, marks_grey

# 3.6 - Update Theme-Mode based on selected theme:
@app.callback(
    [Output('page_content', 'className'),
     Output('current_theme', 'data')],
    [Input('theme_dropdown', 'value')]
)
def update_theme_mode(theme):
    if theme == 'light':
        return 'light_theme', 'light'
    else:
        return 'dark_theme', 'dark'

# 3.7 - Update Dropdown-Classname based on selected theme:
@app.callback(
    Output('city_dropdown', 'className'),
    [Input('current_theme', 'data')]
)
def update_dropdown_classname(current_theme):
    if current_theme == 'light':
        return 'dropdown light_theme_dropdown'
    else:
        return 'dropdown dark_theme_dropdown'

"""
-----------------------------------------------------------------------------------------
Section 4:
4.1 - Definition of KPI-Area
"""
@app.callback(
    Output('table_container', 'children'),
    [Input('city_dropdown', 'value')]
)
def update_table_container(selected_city):
    # Filter data based on the selected city
    if selected_city:
        filtered_df = df[df['CityMap'] == selected_city]
        unique_users_df = filtered_df.drop_duplicates(subset=['user', 'FixationDuration_aggregated'], keep='first')
    # No city is selected
    else:
        filtered_df = df
        unique_users_df = df.drop_duplicates(subset=['user', 'CityMap', 'FixationDuration_aggregated'], keep='first')

    # 1. Average Task Duration (seconds):
    # Sum of FixationDuration per Color / Number of Users per Color
    avg_task_color = unique_users_df[unique_users_df['description'] == 'color']['FixationDuration_aggregated'].mean()
    avg_task_grey = unique_users_df[unique_users_df['description'] == 'grey']['FixationDuration_aggregated'].mean()

    # 2. Number of Fixation-Points (without unit):
    fixation_points_color = filtered_df[filtered_df['description'] == 'color'].shape[0]
    fixation_points_grey = filtered_df[filtered_df['description'] == 'grey'].shape[0]

    # 3. Average Saccade Length (without unit):
    # Lenght of the movement between two fixation points
    avg_saccade_color = filtered_df[filtered_df['description'] == 'color']['SaccadeLength'].mean()
    avg_saccade_grey = filtered_df[filtered_df['description'] == 'grey']['SaccadeLength'].mean()

    # 4. Average Fixation Duration (seconds):
    avg_fixation_duration_color = unique_users_df[unique_users_df['description'] == 'color']['FixationDuration_avg'].mean()
    avg_fixation_duration_grey = unique_users_df[unique_users_df['description'] == 'grey']['FixationDuration_avg'].mean()

    return dash_table.DataTable(
        id='kpi_table',
        columns=[
            {"name": f"KPI for {selected_city or 'all cities'}", "id": "KPI"},
            {"name": "Color Map", "id": "color"},
            {"name": "Greyscale Map", "id": "greyscale"}
        ],
        data=[
            {"KPI": "Avgerage Task Duration",
                "color": f"{avg_task_color:.2f} sec.",
                "greyscale": f"{avg_task_grey:.2f} sec."},
            {"KPI": "Number of Fixation-Points",
                "color": f"{fixation_points_color:,}".replace(',', "'"),
                "greyscale": f"{fixation_points_grey:,}".replace(',', "'")},
            {"KPI": "Avgerage Saccade Length",
                "color": f"{avg_saccade_color:.2f}",
                "greyscale": f"{avg_saccade_grey:.2f}"},
            {"KPI": "Avgerage Fixation Duration",
                "color": f"{avg_fixation_duration_color:.2f} sec.",
                "greyscale": f"{avg_fixation_duration_grey:.2f} sec."}
        ],
        style_cell={
            'textAlign': 'left',
            'padding': '4px',
            'whiteSpace': 'nowrap',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'font': 'normal 10px Arial'
        },
        style_header={
            'backgroundColor': '#000000',
            'color': 'white',
            'textAlign': 'left',
            'padding': '4px',
            'font': 'normal 10px Arial'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'even'},
                'backgroundColor': '#E6E6E6',
                'color': 'black',},
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#CBCBCB',
                'color': 'black',},
            {
                'if': {'column_id': 'KPI'},
                'minWidth': '120px', 'maxWidth': '120px',},
            {
                'if': {'column_id': 'color'},
                'minWidth': '60px', 'maxWidth': '60px',},
            {
                'if': {'column_id': 'greyscale'},
                'minWidth': '60px', 'maxWidth': '60px',},
        ]
    )


"""
-----------------------------------------------------------------------------------------
Section 4:
4.2 - Definition of Scatter-Plot Color (Gaze-Plot)
"""
def get_image_path_color(selected_city):
    file_pattern_color = f'assets/*_{selected_city}_Color.jpg'
    matching_files = glob.glob(file_pattern_color)
    if matching_files:
        image_path = matching_files[0]
        img = Image.open(image_path)
        width, height = img.size
        return 'http://127.0.0.1:8050/' + image_path, width, height
    return None, None, None


@app.callback(
    Output('gaze_plot_color', 'figure'),
    [Input('city_dropdown', 'value'),
     Input('dropdown_user_color', 'value'),
     Input('range_slider_color', 'value'),
     Input('current_theme', 'data')]
)
def update_scatter_plot_color(selected_city, selected_users, range_slider_value, current_theme):
    if selected_city:
        # Define a color map for users
        unique_users = df['user'].dropna().unique()
        colors = px.colors.qualitative.Plotly
        color_map = {user: colors[i % len(colors)] for i, user in enumerate(unique_users)}

        # Filter and sort data based on the selected filters (city and user):
        filtered_df = df[
            (df['CityMap'] == selected_city) & (df['description'] == 'color')]

        if selected_users:
            if isinstance(selected_users, str):
                selected_users = [selected_users]
            filtered_df = filtered_df[filtered_df['user'].isin(selected_users)]

        min_duration, max_duration = range_slider_value
        filtered_df = filtered_df[
            (filtered_df['FixationDuration_aggregated'] >= min_duration) & (filtered_df['FixationDuration_aggregated'] <= max_duration)]

        # Extract Image Information and normalize data (only applicable for Antwerpen):
        image_path_color, width, height = get_image_path_color(selected_city)
        if image_path_color and width and height:
            # Attention: "Antwerpen_S1_Color" Data are not normalized !!!
            if selected_city == 'Antwerpen_S1':
                filtered_df['NormalizedPointX'] = (
                        filtered_df['MappedFixationPointX'] / 1651.00 * width)
                filtered_df['NormalizedPointY'] = (
                        filtered_df['MappedFixationPointY'] / 1200.00 * height)
            else:
                filtered_df['NormalizedPointX'] = filtered_df['MappedFixationPointX']
                filtered_df['NormalizedPointY'] = filtered_df['MappedFixationPointY']

            # Filter for fixation points within map only
            filtered_df = filtered_df[
                (filtered_df['NormalizedPointX'] >= 0) & (filtered_df['NormalizedPointX'] <= width) &
                (filtered_df['NormalizedPointY'] >= 0) & (filtered_df['NormalizedPointY'] <= height)]

        # Set title color based on theme
        title_color = 'black' if current_theme == 'light' else 'white'

        # Create scatter plot using the color map
        fig = px.scatter(filtered_df,
                         x='NormalizedPointX',
                         y='NormalizedPointY',
                         size='FixationDuration',
                         color='user',
                         color_discrete_map=color_map,
                         labels={
                             'MappedFixationPointX': 'X Coordinate',
                             'MappedFixationPointY': 'Y Coordinate',
                             'FixationDuration': 'FixationDuration (ms)',
                             'FixationDuration_aggregated': 'Task Duration (sec)'
                         },
                         hover_data = {
                                'user': True,
                                'MappedFixationPointX': True,
                                'MappedFixationPointY': True,
                                'FixationDuration': True,
                                'FixationDuration_aggregated': True
                            }
                         )

        # Add line traces for each user
        for user in filtered_df['user'].unique():
            user_df = filtered_df[filtered_df['user'] == user]
            fig.add_trace(
                go.Scatter(
                    x=user_df['NormalizedPointX'],
                    y=user_df['NormalizedPointY'],
                    mode='lines',
                    line=dict(width=2, color=color_map[user]),
                    name=f"Scanpath for {user}",
                    hoverinfo='skip'
                )
            )
        fig.update_xaxes(
            range=[0, width],
            autorange=False,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color=title_color, size=9, family='Arial, sans-serif'))

        fig.update_yaxes(
            range=[height, 0],
            autorange=False,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color=title_color, size=9, family='Arial, sans-serif'))

        fig.add_layout_image(
            dict(
                source=image_path_color,
                x=0,
                sizex=width,
                y=0,
                sizey=height,
                xref="x",
                yref="y",
                sizing="stretch",
                opacity=0.8,
                layer="below"
            )
        )

        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            xaxis_title=None,
            yaxis_title=None,
            title={
                'text': f'<b>Color Map Observations in {selected_city}</b>',
                'font': {
                    'size': 12,
                    'family': 'Arial, sans-serif',
                    'color': title_color }
            },
            margin=dict(l=0, r=5, t=40, b=5),
            showlegend=False,
            height=425)
        return fig

    else:
        fig = px.scatter()

        title_color = 'black' if current_theme == 'light' else 'white'

        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            title={'text': f"No City Map selected.<br><br>"
                           f"To display the <b>Scan Path Visualization</b> on a specific map,<br>"
                           f"please select a city from the dropdown on the left.",
                   'y': 0.6,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'middle',
                   'font': dict(
                       size=14,
                       color=title_color,
                       family='Arial, sans-serif')},
            showlegend=False,
            margin=dict(l=0, r=5, t=40, b=5),
            height=425
        )

        fig.update_xaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False)

        return fig


"""
-----------------------------------------------------------------------------------------
Section 4:
4.3 - Definition of Scatter-Plot Grey (Gaze-Plot)
"""
def get_image_path_grey(selected_city):
    file_pattern_grey = f'assets/*_{selected_city}_Grey.jpg'
    matching_files = glob.glob(file_pattern_grey)
    if matching_files:
        image_path = matching_files[0]
        img = Image.open(image_path)
        width, height = img.size
        return 'http://127.0.0.1:8050/' + image_path, width, height
    return None, None, None

@app.callback(
    Output('gaze_plot_grey', 'figure'),
    [Input('city_dropdown', 'value'),
     Input('dropdown_user_grey', 'value'),
     Input('range_slider_grey', 'value'),
     Input('current_theme', 'data')]
)

def update_scatter_plot_grey(selected_city, selected_users, range_slider_value, current_theme):
    if selected_city:
        # Define a color map for users
        unique_users = df['user'].dropna().unique()
        colors = px.colors.qualitative.Plotly
        color_map = {user: colors[i % len(colors)] for i, user in enumerate(unique_users)}

        # Filter and sort data based on the selected filters (city and user):
        filtered_df = df[
            (df['CityMap'] == selected_city) & (df['description'] == 'grey')]

        if selected_users:
            if isinstance(selected_users, str):
                selected_users = [selected_users]
            filtered_df = filtered_df[filtered_df['user'].isin(selected_users)]

        min_duration, max_duration = range_slider_value
        filtered_df = filtered_df[
            (filtered_df['FixationDuration_aggregated'] >= min_duration) & (
                        filtered_df['FixationDuration_aggregated'] <= max_duration)]

        # Extract Image Information:
        image_path_grey, width, height = get_image_path_grey(selected_city)
        if image_path_grey and width and height:
            # Filter for fixation points within map only
            filtered_df = filtered_df[
                (filtered_df['MappedFixationPointX'] >= 0) & (filtered_df['MappedFixationPointX'] <= width) &
                (filtered_df['MappedFixationPointY'] >= 0) & (filtered_df['MappedFixationPointY'] <= height)]

        # Set title color based on theme
        title_color = 'black' if current_theme == 'light' else 'white'

        # Create scatter plot using the color map
        fig = px.scatter(filtered_df,
                         x='MappedFixationPointX',
                         y='MappedFixationPointY',
                         size='FixationDuration',
                         color='user',
                         color_discrete_map=color_map,
                         labels={
                             'MappedFixationPointX': 'X Coordinate',
                             'MappedFixationPointY': 'Y Coordinate',
                             'FixationDuration': 'FixationDuration (ms)',
                             'FixationDuration_aggregated': 'Task Duration (sec)'
                         },
                         hover_data = {
                                'user': True,
                                'MappedFixationPointX': True,
                                'MappedFixationPointY': True,
                                'FixationDuration': True,
                                'FixationDuration_aggregated': True
                            }
        )

        # Add line traces for each user
        for user in filtered_df['user'].unique():
            user_df = filtered_df[filtered_df['user'] == user]
            fig.add_trace(
                go.Scatter(
                    x=user_df['MappedFixationPointX'],
                    y=user_df['MappedFixationPointY'],
                    mode='lines',
                    line=dict(width=2, color=color_map[user]),
                    name=f"Scanpath for {user}",
                    hoverinfo='skip'
                )
            )

        fig.update_xaxes(
            range=[0, width],
            autorange=False,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color=title_color, size=9, family='Arial, sans-serif'))

        fig.update_yaxes(
            range=[height, 0],
            autorange=False,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color=title_color, size=9, family='Arial, sans-serif'))

        # Add Background Image
        fig.add_layout_image(
            dict(
                source=image_path_grey,
                x=0,
                sizex=width,
                y=0,
                sizey=height,
                xref="x",
                yref="y",
                sizing="stretch",
                opacity=0.8,
                layer="below"
            )
        )

        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Set background color to transparent
            paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper color to transparent
            xaxis_title=None,
            yaxis_title=None,
            title={
                'text': f'<b>Greyscale Map Observations in {selected_city}</b>',
                'font': {
                    'size': 12,
                    'family': 'Arial, sans-serif',
                    'color': title_color }
            },
            margin=dict(l=0, r=5, t=40, b=5),
            showlegend=False,
            height=425)
        return fig

    else:
        title_color = 'black' if current_theme == 'light' else 'white'
        cities = [
            {"name": "Antwerpen", "lat": 51.2194, "lon": 4.4025},
            {"name": "Berlin", "lat": 52.5200, "lon": 13.4050},
            {"name": "Bordeaux", "lat": 44.8378, "lon": -0.5792},
            {"name": "Köln", "lat": 50.9375, "lon": 6.9603},
            {"name": "Frankfurt", "lat": 50.1109, "lon": 8.6821},
            {"name": "Hamburg", "lat": 53.5511, "lon": 9.9937},
            {"name": "Moskau", "lat": 55.7558, "lon": 37.6173},
            {"name": "Riga", "lat": 56.9496, "lon": 24.1052},
            {"name": "Tokyo", "lat": 35.6895, "lon": 139.6917},
            {"name": "Barcelona", "lat": 41.3851, "lon": 2.1734},
            {"name": "Bologna", "lat": 44.4949, "lon": 11.3426},
            {"name": "Brüssel", "lat": 50.8503, "lon": 4.3517},
            {"name": "Budapest", "lat": 47.4979, "lon": 19.0402},
            {"name": "Düsseldorf", "lat": 51.2277, "lon": 6.7735},
            {"name": "Göteborg", "lat": 57.7089, "lon": 11.9746},
            {"name": "Hong-Kong", "lat": 22.3193, "lon": 114.1694},
            {"name": "Krakau", "lat": 50.0647, "lon": 19.9450},
            {"name": "Ljubljana", "lat": 46.0569, "lon": 14.5058},
            {"name": "New-York", "lat": 40.7128, "lon": -74.0060},
            {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
            {"name": "Pisa", "lat": 43.7228, "lon": 10.4017},
            {"name": "Venedig", "lat": 45.4408, "lon": 12.3155},
            {"name": "Warschau", "lat": 52.2297, "lon": 21.0122},
            {"name": "Zürich", "lat": 47.3769, "lon": 8.5417}
        ]

        lats = [city["lat"] for city in cities]
        lons = [city["lon"] for city in cities]
        names = [city["name"] for city in cities]

        fig = go.Figure()

        fig.add_trace(go.Scattergeo(
            locationmode='ISO-3',
            lon=lons,
            lat=lats,
            text=names,
            mode='markers',
            marker=dict(
                size=6,
                symbol='circle',
                color='blue'
            ),
            textposition='top right',
            hoverinfo='text'
        ))

        fig.update_layout(
            title=dict(
                text='<br><br><b>Available City Maps</b><br>'
                     '(zoom out to see cities outside Europe)',
                font=dict(size=12, family='Arial, sans-serif', color=title_color),
            ),
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='lightgray',
                coastlinecolor='darkgray',
                showcoastlines=True,
                showcountries=True,
                countrycolor='darkgray',
                lonaxis=dict(range=[-10, 40]),  # Longitude range for Europe
                lataxis=dict(range=[35, 65])  # Latitude range for Europe
            ),
            margin=dict(l=5, r=5, t=100, b=5),
            height=424
        )
        return fig

"""
-----------------------------------------------------------------------------------------
Section 4:
4.4 - Definition of Density Heat-Map Color
"""
@app.callback(
    Output('heat_map_color', 'figure'),
    [Input('city_dropdown', 'value'),
     Input('dropdown_user_color', 'value'),
     Input('range_slider_color', 'value'),
     Input('current_theme', 'data')]
)

def update_heatmap_color(selected_city, selected_users, range_slider_value, current_theme):
    if selected_city:
        # Filter and sort data based on the selected filters (city and user):
        filtered_df = df[(df['CityMap'] == selected_city) & (df['description'] == 'color')]

        if selected_users:
            if isinstance(selected_users, str):
                selected_users = [selected_users]
            filtered_df = filtered_df[filtered_df['user'].isin(selected_users)]

        min_duration, max_duration = range_slider_value
        filtered_df = filtered_df[
            (filtered_df['FixationDuration_aggregated'] >= min_duration) & (
                        filtered_df['FixationDuration_aggregated'] <= max_duration)]

        # Extract Image Information and normalize data (only applicable for Antwerpen):
        image_path_color, width, height = get_image_path_color(selected_city)
        if image_path_color and width and height:
            # Attention: "Antwerpen_S1_Color" Data are not normalized !!!
            if selected_city == 'Antwerpen_S1':
                filtered_df['NormalizedPointX'] = (
                        filtered_df['MappedFixationPointX'] / 1651.00 * width)
                filtered_df['NormalizedPointY'] = (
                        filtered_df['MappedFixationPointY'] / 1200.00 * height)
            else:
                filtered_df['NormalizedPointX'] = filtered_df['MappedFixationPointX']
                filtered_df['NormalizedPointY'] = filtered_df['MappedFixationPointY']

            # Filter for fixation points within map only
            filtered_df = filtered_df[
                (filtered_df['NormalizedPointX'] >= 0) & (filtered_df['NormalizedPointX'] <= width) &
                (filtered_df['NormalizedPointY'] >= 0) & (filtered_df['NormalizedPointY'] <= height)]

        # Set title color based on theme
        title_color = 'black' if current_theme == 'light' else 'white'

        fig = px.density_contour(filtered_df,
                                 x='NormalizedPointX',
                                 y='NormalizedPointY',
                                 nbinsx=30,
                                 nbinsy=30)

        fig.update_traces(
            contours_showlabels=False,
            contours_coloring="fill",
            line=dict(
                smoothing=1.3,
                color='rgba(0, 0, 0, 0)'
            ),
            colorscale=[
                [0.0, "rgba(0, 128, 0, 0)"],  # Green, but transparent
                [0.2, "rgba(0, 128, 0, 0.5)"],  # Green with some opacity
                [0.4, "rgba(173, 255, 47, 0.6)"],  # Yellow-green with moderate opacity
                [0.6, "rgba(255, 255, 0, 0.7)"],  # Yellow with higher opacity
                [0.8, "rgba(255, 165, 0, 0.8)"],  # Orange with more opacity
                [1.0, "rgba(255, 0, 0, 0.9)"]  # Red with full opacity
            ],
            showscale=False)

        fig.update_xaxes(
            range=[0, width],
            autorange=False,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color=title_color, size=9, family='Arial, sans-serif'))

        fig.update_yaxes(
            range=[height, 0],
            autorange=False,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color=title_color, size=9, family='Arial, sans-serif'))

        fig.add_layout_image(
            dict(
                source=image_path_color,
                x=0,
                sizex=width,
                y=0,
                sizey=height,
                xref="x",
                yref="y",
                sizing="stretch",
                opacity=0.8,
                layer="below"
            )
        )

        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            xaxis_title=None,
            yaxis_title=None,
            title={
                'text': f'<b>Color Map Observations in {selected_city}</b>',
                'font': {
                    'size': 12,
                    'family': 'Arial, sans-serif',
                    'color': title_color}
            },
            margin=dict(l=0, r=5, t=40, b=5),
            showlegend=False,
            height=425)
        return fig

    else:
        fig = px.scatter()

        title_color = 'black' if current_theme == 'light' else 'white'

        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            title={
                'text': f"No City Map selected.<br><br>"
                        f"To display the <b>Density Visualization</b> on a specific map,<br>"
                        f"please select a city from the dropdown on the left.",
                'y': 0.6,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'middle',
                'font': {
                    'size': 14,
                    'color': title_color,
                    'family': 'Arial, sans-serif'
                }},
            showlegend=False,
            margin=dict(l=0, r=5, t=40, b=5),
            height=425,
        )

        fig.update_xaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False)

        return fig

"""
-----------------------------------------------------------------------------------------
Section 4:
4.5 - Definition of Density Heat-Map Grey
"""
@app.callback(
    Output('heat_map_grey', 'figure'),
    [Input('city_dropdown', 'value'),
     Input('dropdown_user_grey', 'value'),
     Input('range_slider_grey', 'value'),
     Input('current_theme', 'data')]
)

def update_heatmap_grey(selected_city, selected_users, range_slider_value, current_theme):
    if selected_city:
        # Filter and sort data based on the selected filters (city and user):
        filtered_df = df[(df['CityMap'] == selected_city) & (df['description'] == 'grey')]

        if selected_users:
            if isinstance(selected_users, str):
                selected_users = [selected_users]
            filtered_df = filtered_df[filtered_df['user'].isin(selected_users)]

        min_duration, max_duration = range_slider_value
        filtered_df = filtered_df[
            (filtered_df['FixationDuration_aggregated'] >= min_duration) & (
                    filtered_df['FixationDuration_aggregated'] <= max_duration)]

        # Extract Image Information:
        image_path_grey, width, height = get_image_path_grey(selected_city)
        if image_path_grey and width and height:
            # Filter for fixation points within map only
            filtered_df = filtered_df[
                (filtered_df['MappedFixationPointX'] >= 0) & (filtered_df['MappedFixationPointX'] <= width) &
                (filtered_df['MappedFixationPointY'] >= 0) & (filtered_df['MappedFixationPointY'] <= height)]

        # Set title color based on theme
        title_color = 'black' if current_theme == 'light' else 'white'

        fig = px.density_contour(filtered_df,
                                 x='MappedFixationPointX',
                                 y='MappedFixationPointY',
                                 nbinsx=30,
                                 nbinsy=30)

        fig.update_traces(
            contours_showlabels=False,
            contours_coloring="fill",
            line=dict(
                smoothing=1.3,
                color='rgba(0, 0, 0, 0)'  # Set contour line color to transparent
            ),
            colorscale=[
                [0.0, "rgba(0, 128, 0, 0)"],  # Green, but transparent
                [0.2, "rgba(0, 128, 0, 0.5)"],  # Green with some opacity
                [0.4, "rgba(173, 255, 47, 0.6)"],  # Yellow-green with moderate opacity
                [0.6, "rgba(255, 255, 0, 0.7)"],  # Yellow with higher opacity
                [0.8, "rgba(255, 165, 0, 0.8)"],  # Orange with more opacity
                [1.0, "rgba(255, 0, 0, 0.9)"]  # Red with full opacity
            ],
            showscale=False)

        fig.update_xaxes(
            range=[0, width],
            autorange=False,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color=title_color, size=9, family='Arial, sans-serif'))

        fig.update_yaxes(
            range=[height, 0],
            autorange=False,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color=title_color, size=9, family='Arial, sans-serif'))

        fig.add_layout_image(
            dict(
                source=image_path_grey,
                x=0,
                sizex=width,
                y=0,
                sizey=height,
                xref="x",
                yref="y",
                sizing="stretch",
                opacity=0.8,
                layer="below"
            )
        )

        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            xaxis_title=None,
            yaxis_title=None,
            title={
                'text': f'<b>Greyscale Map Observations in {selected_city}</b>',
                'font': {
                    'size': 12,
                    'family': 'Arial, sans-serif',
                    'color': title_color}
            },
            margin=dict(l=0, r=5, t=40, b=5),
            showlegend=False,
            height=425,)
        return fig


    else:
        title_color = 'black' if current_theme == 'light' else 'white'

        cities = [
            {"name": "Antwerpen", "lat": 51.2194, "lon": 4.4025},
            {"name": "Berlin", "lat": 52.5200, "lon": 13.4050},
            {"name": "Bordeaux", "lat": 44.8378, "lon": -0.5792},
            {"name": "Köln", "lat": 50.9375, "lon": 6.9603},
            {"name": "Frankfurt", "lat": 50.1109, "lon": 8.6821},
            {"name": "Hamburg", "lat": 53.5511, "lon": 9.9937},
            {"name": "Moskau", "lat": 55.7558, "lon": 37.6173},
            {"name": "Riga", "lat": 56.9496, "lon": 24.1052},
            {"name": "Tokyo", "lat": 35.6895, "lon": 139.6917},
            {"name": "Barcelona", "lat": 41.3851, "lon": 2.1734},
            {"name": "Bologna", "lat": 44.4949, "lon": 11.3426},
            {"name": "Brüssel", "lat": 50.8503, "lon": 4.3517},
            {"name": "Budapest", "lat": 47.4979, "lon": 19.0402},
            {"name": "Düsseldorf", "lat": 51.2277, "lon": 6.7735},
            {"name": "Göteborg", "lat": 57.7089, "lon": 11.9746},
            {"name": "Hong-Kong", "lat": 22.3193, "lon": 114.1694},
            {"name": "Krakau", "lat": 50.0647, "lon": 19.9450},
            {"name": "Ljubljana", "lat": 46.0569, "lon": 14.5058},
            {"name": "New-York", "lat": 40.7128, "lon": -74.0060},
            {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
            {"name": "Pisa", "lat": 43.7228, "lon": 10.4017},
            {"name": "Venedig", "lat": 45.4408, "lon": 12.3155},
            {"name": "Warschau", "lat": 52.2297, "lon": 21.0122},
            {"name": "Zürich", "lat": 47.3769, "lon": 8.5417}
        ]

        lats = [city["lat"] for city in cities]
        lons = [city["lon"] for city in cities]
        names = [city["name"] for city in cities]
        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            locationmode='ISO-3',
            lon=lons,
            lat=lats,
            text=names,
            mode='markers',
            marker=dict(
                size=6,
                symbol='circle',
                color='blue'
            ),
            textposition='top right',
            hoverinfo='text'
        ))

        fig.update_layout(
            title=dict(
                text='<br><br><b>Available City Maps</b><br>'
                     '(zoom out to see cities outside Europe)',
                font=dict(size=12, family='Arial, sans-serif', color=title_color),
            ),
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='lightgray',
                coastlinecolor='darkgray',
                showcoastlines=True,
                showcountries=True,
                countrycolor='darkgray',
                lonaxis=dict(range=[-10, 40]),  # Longitude range for Europe
                lataxis=dict(range=[35, 65])  # Latitude range for Europe
            ),
            margin=dict(l=5, r=5, t=100, b=5),
            height=424
        )

        return fig

"""
-----------------------------------------------------------------------------------------
Section 4:
4.6 - Definition of Box-Plot "Task Duration" (Distribution of Task Duration (A-B) per User, Color, City)
"""
@app.callback(
    Output('box_task_duration', 'figure'),
    [Input('active-button', 'data'),
     Input('current_theme', 'data')]
)
def update_box_plot_task_duration(active_button, current_theme):
    if active_button == 'default_viz':
        city_order = sorted(df['City'].unique().tolist())

        # Set title color based on theme
        title_color = 'black' if current_theme == 'light' else 'white'

        # Calculate medians for annotations
        medians = df.groupby(['City', 'description'])['FixationDuration_aggregated'].median().reset_index()
        max_fixation_duration = df['FixationDuration_aggregated'].max()

        fig = px.box(df,
                     x='FixationDuration_aggregated',
                     y='City',
                     points=False,
                     color='description',
                     boxmode='group',
                     category_orders={'City': city_order},
                     color_discrete_map={
                         'color': 'blue',
                         'grey': 'lightgrey'},
                     labels={'FixationDuration_aggregated': 'Task Duration [sec.]',
                             'City': '',
                             'description': ''})

        fig.update_traces(marker=dict(size=8), line=dict(width=2.0))

        fig.update_xaxes(dtick=100,
                         showticklabels=True,
                         tickfont=dict(color=title_color, size=11, family='Arial, sans-serif'),
                         showgrid=False,
                         showline=True,
                         zeroline=False,
                         linecolor=title_color,
                         linewidth=0.2)

        fig.update_yaxes(dtick=1,
                         showgrid=False,
                         showticklabels=True,
                         zeroline=False,
                         showline=False,
                         tickfont=dict(color=title_color, size=11, family='Arial, sans-serif'))

        # Add median text annotations aligned along the right edge
        x_offsets = {
            'color': max_fixation_duration * 1.05,  # Slightly outside the max x value
            'grey': max_fixation_duration * 1.15  # Further outside to avoid overlap
        }

        for description in medians['description'].unique():
            median_data = medians[medians['description'] == description]
            for _, row in median_data.iterrows():
                x_value = x_offsets[description]  # Adjusted position
                y_value = row['City']
                color = 'grey' if description == 'grey' else 'blue'

                fig.add_annotation(
                    x=x_value,
                    y=y_value,
                    text=f'{row["FixationDuration_aggregated"]:.2f}',
                    showarrow=False,
                    xanchor='left',
                    yanchor='middle',
                    font=dict(size=9, color=color)
                )

        fig.update_layout(
            height=525,
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            yaxis_title=None,
            xaxis_title={
                'text': 'Task Duration [sec.]',
                'font': {
                    'size': 11,
                    'family': 'Arial, sans-serif',
                    'color': title_color}
            },
            title={
                'text': f'<b>Distribution of Task-Duration</b>'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i>Median:</i>',
                'font': {
                    'size': 12,
                    'family': 'Arial, sans-serif',
                    'color': title_color}
            },
            margin=dict(l=5, r=5, t=40, b=5),
            legend=dict(
                font=dict(color=title_color, size=10, family='Arial, sans-serif'),
                bgcolor='rgba(0, 0, 0, 0)',
                orientation='h',
                yanchor='top',
                y=-0.04,
                xanchor='left',
                x=-0.2
            ),
            showlegend=True,
            )

        return fig

    else:
        fig = px.scatter()
        return fig

"""
-----------------------------------------------------------------------------------------
Section 4:
4.7 - Definition of Box-Plot "Average Fixation Duration" (Distribution of Avg. Fixation Duration per User, Color, City)
"""
@app.callback(
    Output('box_avg_fix_duration', 'figure'),
    [Input('active-button', 'data'),
     Input('current_theme', 'data')]
)
def update_box_plot_avg_fix_duration(active_button, current_theme):
    if active_button == 'default_viz':
        city_order = sorted(df['City'].unique().tolist())

        # Set title color based on theme
        title_color = 'black' if current_theme == 'light' else 'white'

        # Calculate medians for annotations
        medians = df.groupby(['City', 'description'])['FixationDuration_avg'].median().reset_index()
        max_fixation_duration = df['FixationDuration_avg'].max()

        fig = px.box(df,
                     x='FixationDuration_avg',
                     y='City',
                     points=False,
                     color='description',
                     boxmode='group',
                     category_orders={'City': city_order},
                     color_discrete_map={
                         'color': 'blue',
                         'grey': 'lightgrey'},
                     labels={'FixationDuration_avg': 'Avg. Fixation Duration [sec.]',
                             'City': '',
                             'description': ''})

        fig.update_traces(marker=dict(size=8), line=dict(width=2.0))

        fig.update_xaxes(dtick=0.2,
                         showticklabels=True,
                         tickfont=dict(color=title_color, size=11, family='Arial, sans-serif'),
                         showgrid=False,
                         showline=True,
                         zeroline=False,
                         linecolor=title_color,
                         linewidth=0.2)

        fig.update_yaxes(dtick=1,
                         showgrid=False,
                         showticklabels=True,
                         zeroline=False,
                         showline=False,
                         tickfont=dict(color=title_color, size=11, family='Arial, sans-serif')
                         )

        # Add median text annotations aligned along the right edge
        x_offsets = {
            'color': max_fixation_duration * 1.05,  # Slightly outside the max x value
            'grey': max_fixation_duration * 1.15  # Further outside to avoid overlap
        }

        for description in medians['description'].unique():
            median_data = medians[medians['description'] == description]
            for _, row in median_data.iterrows():
                x_value = x_offsets[description]  # Adjusted position
                y_value = row['City']
                color = 'grey' if description == 'grey' else 'blue'

                fig.add_annotation(
                    x=x_value,
                    y=y_value,
                    text=f'{row["FixationDuration_avg"]:.2f}',
                    showarrow=False,
                    xanchor='left',
                    yanchor='middle',
                    font=dict(size=9, color=color)
                )

        fig.update_layout(
            height=525,
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            yaxis_title=None,
            xaxis_title={
                'text': 'Avg. Fixation Duration [sec.]',
                'font': {
                    'size': 11,
                    'family': 'Arial, sans-serif',
                    'color': title_color}
            },
            title={
                'text': f'<b>Distribution of Average Fixation Duration</b>'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                        f'&nbsp;&nbsp;<i>Median:</i>',
                'font': {
                    'size': 12,
                    'family': 'Arial, sans-serif',
                    'color': title_color}
            },
            margin=dict(l=5, r=5, t=40, b=5),
            legend=dict(
                font=dict(color=title_color, size=10, family='Arial, sans-serif'),
                bgcolor='rgba(0, 0, 0, 0)',
                orientation='h',
                yanchor='top',
                y=-0.04,
                xanchor='left',
                x=-0.2
            ),
            showlegend=True,
        )

        return fig

    else:
        fig = px.scatter()
        return fig

"""
-----------------------------------------------------------------------------------------
Section 4:
4.8 - Definition of Histogram (Distribution of Task Duration per selected city map)
"""
@app.callback(
    Output('hist_taskduration', 'figure'),
     [Input('city_dropdown', 'value'),
     Input('current_theme', 'data')]
)
def update_histogram_task_duration(selected_city, current_theme):
    title_color = 'black' if current_theme == 'light' else 'white'

    if selected_city:
        filtered_df = df[df['CityMap'] == selected_city]
        unique_users_df = filtered_df.drop_duplicates(subset=['user', 'FixationDuration_aggregated'], keep='first')
        titel = (f'<b>Distribution of Task Duration in {selected_city}</b><br><br>'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;color'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;grey')

        fig = px.histogram(unique_users_df,
                           x="FixationDuration_aggregated",
                           color="description",
                           facet_col='description',
                           category_orders={"description": ["color", "grey"]},
                           color_discrete_map={
                               'color': 'blue',
                               'grey': 'lightgrey'},
                           nbins=20,
                           labels={
                               "FixationDuration_aggregated": ""
                           })

    else:
        unique_users_df = df.drop_duplicates(subset=['user', 'CityMap', 'FixationDuration_aggregated'], keep='first')
        titel = (f'<b>Distribution of Task Duration in all cities</b><br><br>'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;color'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                 f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;grey')

        fig = px.histogram(unique_users_df,
                           x="FixationDuration_aggregated",
                           color="description",
                           facet_col='description',
                           category_orders={"description": ["color", "grey"]},
                           color_discrete_map={
                               'color': 'blue',
                               'grey': 'lightgrey'},
                           nbins=50,
                           labels={
                               "FixationDuration_aggregated": ""
                           })

    fig.update_xaxes(showgrid=False,
                     showticklabels=True,
                     tickfont=dict(color=title_color, size=11, family='Arial, sans-serif'),
                     showline=True,
                     linecolor=title_color,
                     zeroline=False)

    fig.update_yaxes(showgrid=False,
                     showticklabels=False,
                     tickfont=dict(color=title_color, size=11, family='Arial, sans-serif'),
                     showline=True,
                     linecolor=title_color,
                     zeroline=False)

    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        yaxis_title=None,
        xaxis_title=dict(
            text='[sec.]',
            font=dict(size=10, family='Arial, sans-serif', color=title_color)),
        margin=dict(l=1, r=5, t=30, b=0),
        showlegend=False,
        title=dict(
            text=titel,
            font=dict(size=12, family='Arial, sans-serif', color=title_color)
        ),
        legend_title_text='',
        height=137)

    fig.update_traces(
        marker_line_color=title_color,
        marker_line_width=1
    )

    fig.for_each_annotation(lambda a: a.update(text=''))
    fig.for_each_xaxis(lambda axis: axis.update(
        title_text='[sec.]',
        title_font=dict(size=10, family='Arial, sans-serif', color=title_color))
                       )
    return fig

"""
-----------------------------------------------------------------------------------------
Section 4:
4.9 - Definition of Scatter Plot Color (Correlation between Fixation Duration and Saccade Length)
"""
@app.callback(
    Output('scatter_correlation_color', 'figure'),
    [Input('active-button', 'data'),
     Input('city_dropdown', 'value'),
     Input('current_theme', 'data')]
)
def update_scatter_correlation_color(active_button, selected_city, current_theme):
    if active_button == 'scatter_plot':
        # Set title color based on theme
        title_color = 'black' if current_theme == 'light' else 'white'

        if selected_city:
            filtered_df = df[(df['CityMap'] == selected_city) & (df['description'] == 'color')]
            title = (f'<b>Color Map {selected_city}:<br>'
                     f'Correlation between Saccade Length and Fixation Duration</b>')

            # Drop data where 'SaccadeLength' is null
            filtered_df = filtered_df.dropna(subset=['FixationDuration', 'SaccadeLength'])

            # Convert FixationDuration to seconds
            filtered_df['FixationDuration'] = filtered_df['FixationDuration'] / 1000

            min_x = filtered_df['FixationDuration'].min()
            max_x = filtered_df['FixationDuration'].max()
            min_y = filtered_df['SaccadeLength'].min()
            max_y = filtered_df['SaccadeLength'].max()

            color_sequence = ['rgba(94, 204, 244, 1)', 'rgba(0, 0, 255, 1)']

            fig = px.scatter(filtered_df,
                             x='FixationDuration',
                             y='SaccadeLength',
                             category_orders={'TaskDurationCategory': ['<10 sec.', '>=10 sec.']},
                             color='TaskDurationCategory',
                             color_discrete_sequence=color_sequence,
                             labels={'FixationDuration': 'Fixation Duration [sec.]',
                                     'SaccadeLength': 'Saccade Length',
                                     'TaskDurationCategory': 'Task Duration Category'})

            fig.update_traces(marker=dict(size=9,
                                          opacity=0.8,
                                          line=dict(width=0.3, color=title_color))
                              )

            fig.update_layout(
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                margin=dict(l=50, r=30, t=50, b=50),
                height=525,
                title=dict(text=title,
                           font=dict(size=12, family='Arial, sans-serif', color=title_color)),
                yaxis_title=dict(text='Saccade Length',
                                 font=dict(size=11, family='Arial, sans-serif', color=title_color),
                                 standoff=0),
                xaxis_title=dict(text='Fixation Duration [sec.]',
                                 font=dict(size=11, family='Arial, sans-serif', color=title_color)),
                legend_title=dict(text='Task Duration Category',
                                  font=dict(size=10, family='Arial, sans-serif', color=title_color)),
                legend=dict(orientation='h',
                            font=dict(size=10, family='Arial, sans-serif', color=title_color))
            )

            fig.update_xaxes(range=[min_x-0.07, max_x+0.09],
                             showgrid=False,
                             showticklabels=True,
                             tickfont=dict(color=title_color, size=10, family='Arial, sans-serif'),
                             showline=True,
                             zeroline=False,
                             linecolor=title_color,
                             zerolinewidth=0.2)

            fig.update_yaxes(range=[min_y-20, max_y+20],
                             showgrid=False,
                             showticklabels=True,
                             tickfont=dict(color=title_color, size=10, family='Arial, sans-serif'),
                             showline=True,
                             zeroline=False,
                             linecolor=title_color,
                             linewidth=0.2)

            return fig

        else:
            filtered_df = df[df['description'] == 'color']

            # Drop data where 'SaccadeLength' is null
            filtered_df = filtered_df.dropna(subset=['FixationDuration', 'SaccadeLength'])

            # Convert FixationDuration to seconds
            filtered_df['FixationDuration'] = filtered_df['FixationDuration'] / 1000

            # Create a histogram2dcontour plot
            fig = go.Figure(go.Histogram2dContour(
                x=filtered_df['FixationDuration'],
                y=filtered_df['SaccadeLength'],
                coloraxis='coloraxis')
            )

            fig.update_layout(
                coloraxis=dict(
                    colorscale='Blues',
                    colorbar=dict(
                        title='Scatter Density',
                        tickfont=dict(color=title_color, size=10, family='Arial, sans-serif'),
                        titlefont=dict(color=title_color, size=10, family='Arial, sans-serif')))
            )

            fig.update_layout(
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                margin=dict(l=50, r=30, t=50, b=50),
                height=525,
                title=dict(text=f'<b>Color Maps of all cities:<br>'
                           f'Correlation between Saccade Length and Fixation Duration</b>',
                           font=dict(size=12, family='Arial, sans-serif', color=title_color)),
                xaxis_title=dict(text='Fixation Duration [sec.]',
                                 font=dict(size=11, family='Arial, sans-serif', color=title_color)),
                yaxis_title=dict(text='Saccade Length',
                                 font=dict(size=11, family='Arial, sans-serif', color=title_color))
            )

            fig.update_xaxes(range=[0.07, 0.6],
                             showgrid=False,
                             showticklabels=True,
                             tickfont=dict(color=title_color, size=10, family='Arial, sans-serif'),
                             showline=False,
                             zeroline=False,
                             zerolinecolor=title_color,
                             linecolor=title_color,
                             zerolinewidth=0.2)

            fig.update_yaxes(range= [0, 350],
                             showgrid=False,
                             showticklabels=True,
                             tickfont=dict(color=title_color, size=10, family='Arial, sans-serif'),
                             showline=False,
                             zeroline=False,
                             zerolinecolor=title_color,
                             linecolor=title_color,
                             zerolinewidth=0.2)

            return fig

    else:
        fig = px.scatter()
        return fig

"""
-----------------------------------------------------------------------------------------
Section 4:
4.10 - Definition of Scatter Plot Grey (Correlation between Fixation Duration and Saccade Length)
"""
@app.callback(
    Output('scatter_correlation_grey', 'figure'),
    [Input('active-button', 'data'),
     Input('city_dropdown', 'value'),
    Input('current_theme', 'data')]
)
def update_scatter_correlation_grey(active_button, selected_city, current_theme):
    if active_button == 'scatter_plot':
        # Set title color based on theme
        title_color = 'black' if current_theme == 'light' else 'white'

        if selected_city:
            filtered_df = df[(df['CityMap'] == selected_city) & (df['description'] == 'grey')]
            title = (f'<b>Greyscale Map {selected_city}:<br>'
                     f'Correlation between Saccade Length and Fixation Duration</b>')

            # Drop data where 'SaccadeLength' is null
            filtered_df = filtered_df.dropna(subset=['FixationDuration', 'SaccadeLength'])

            # Convert FixationDuration to seconds
            filtered_df['FixationDuration'] = filtered_df['FixationDuration'] / 1000

            min_x = filtered_df['FixationDuration'].min()
            max_x = filtered_df['FixationDuration'].max()
            min_y = filtered_df['SaccadeLength'].min()
            max_y = filtered_df['SaccadeLength'].max()

            color_sequence = ['#333333', 'grey']

            fig = px.scatter(filtered_df,
                             x='FixationDuration',
                             y='SaccadeLength',
                             category_orders={'TaskDurationCategory': ['<10 sec.', '>=10 sec.']},
                             color='TaskDurationCategory',
                             color_discrete_sequence=color_sequence,
                             labels={'FixationDuration': 'Fixation Duration [sec.]',
                                     'SaccadeLength': 'Saccade Length',
                                     'TaskDurationCategory': 'Task Duration Category'})

            fig.update_traces(marker=dict(size=9,
                                          opacity=0.8,
                                          line=dict(width=0.3, color=title_color))
                              )

            fig.update_layout(
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                margin=dict(l=50, r=30, t=50, b=50),
                height=525,
                title=dict(text=title,
                           font=dict(size=12, family='Arial, sans-serif', color=title_color)),
                yaxis_title=dict(text='Saccade Length',
                                 font=dict(size=11, family='Arial, sans-serif', color=title_color),
                                 standoff=0),
                xaxis_title=dict(text='Fixation Duration [sec.]',
                                 font=dict(size=11, family='Arial, sans-serif', color=title_color)),
                legend_title=dict(text='Task Duration Category',
                                  font=dict(size=10, family='Arial, sans-serif', color=title_color)),
                legend=dict(orientation='h',
                            font=dict(size=10, family='Arial, sans-serif', color=title_color))
            )

            fig.update_xaxes(range=[min_x-0.07, max_x+0.09],
                             showgrid=False,
                             showticklabels=True,
                             tickfont=dict(color=title_color, size=10, family='Arial, sans-serif'),
                             showline=True,
                             zeroline=False,
                             linecolor=title_color,
                             zerolinewidth=0.2)

            fig.update_yaxes(range=[min_y-20, max_y+20],
                             showgrid=False,
                             showticklabels=True,
                             tickfont=dict(color=title_color, size=10, family='Arial, sans-serif'),
                             showline=True,
                             zeroline=False,
                             linecolor=title_color,
                             linewidth=0.2)

            return fig

        else:
            filtered_df = df[df['description'] == 'grey']

            # Drop data where 'SaccadeLength' is null
            filtered_df = filtered_df.dropna(subset=['FixationDuration', 'SaccadeLength'])

            # Convert FixationDuration to seconds
            filtered_df['FixationDuration'] = filtered_df['FixationDuration'] / 1000

            # Create a histogram2dcontour plot
            fig = go.Figure(go.Histogram2dContour(
                x=filtered_df['FixationDuration'],
                y=filtered_df['SaccadeLength'],
                coloraxis='coloraxis')
            )

            fig.update_layout(
                coloraxis=dict(
                    colorscale='Greys',
                    colorbar=dict(
                        title='Scatter Density',
                        tickfont=dict(color=title_color, size=10, family='Arial, sans-serif'),
                        titlefont=dict(color=title_color, size=10, family='Arial, sans-serif')))
            )

            fig.update_layout(
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                margin=dict(l=50, r=30, t=50, b=50),
                height=525,
                title=dict(text=f'<b>Greyscale Maps of all cities:<br>'
                           f'Correlation between Saccade Length and Fixation Duration</b>',
                           font=dict(size=12, family='Arial, sans-serif', color=title_color)),
                xaxis_title=dict(text='Fixation Duration [sec.]',
                                 font=dict(size=11, family='Arial, sans-serif', color=title_color)),
                yaxis_title=dict(text='Saccade Length',
                                 font=dict(size=11, family='Arial, sans-serif', color=title_color))
            )

            fig.update_xaxes(range=[0.07, 0.6],
                             showgrid=False,
                             showticklabels=True,
                             tickfont=dict(color=title_color, size=10, family='Arial, sans-serif'),
                             showline=False,
                             zeroline=True,
                             zerolinecolor=title_color,
                             linecolor=title_color,
                             zerolinewidth=0.2)

            fig.update_yaxes(range= [0, 350],
                             showgrid=False,
                             showticklabels=True,
                             tickfont=dict(color=title_color, size=10, family='Arial, sans-serif'),
                             showline=False,
                             zeroline=False,
                             zerolinecolor=title_color,
                             linecolor=title_color,
                             zerolinewidth=0.2)

            return fig

    else:
        fig = px.scatter()
        return fig

if __name__ == '__main__':
    app.run_server(debug=True)
