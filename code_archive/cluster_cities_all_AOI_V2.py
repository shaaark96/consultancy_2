import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import numpy as np

# Daten laden und vorbereiten
data_path = '../assets/all_fixation_data_cleaned_up.csv'
df = pd.read_csv(data_path, sep=';')

# Hinzufügen von "Task Duration in sec" zum DataFrame
task_duration = df.groupby(['user', 'CityMap', 'description'])['FixationDuration'].sum().reset_index()
task_duration['FixationDuration'] = task_duration['FixationDuration'] / 1000
df = pd.merge(df, task_duration, on=['user', 'CityMap', 'description'], suffixes=('', '_aggregated'))

# Hinzufügen von "Average Fixation Duration in sec" zum DataFrame
avg_fix_duration = df.groupby(['user', 'CityMap', 'description'])['FixationDuration'].mean().reset_index()
avg_fix_duration['FixationDuration'] = avg_fix_duration['FixationDuration'] / 1000
df = pd.merge(df, avg_fix_duration, on=['user', 'CityMap', 'description'], suffixes=('', '_avg'))

# Hinzufügen einer Kategorie für die Dauer der Aufgaben
df['TaskDurationCategory'] = pd.cut(df['FixationDuration_aggregated'], bins=[0, 10, float('inf')],
                                    labels=['<10 sec.', '>=10 sec.'])

# Initialisierung der Dash-Anwendung
app = dash.Dash(__name__)

# Layout der Anwendung
app.layout = html.Div([
    html.H1("AOI-Visualisierung mit Clusteranalyse"),

    # Dropdowns zur Auswahl von User und CityMap
    html.Label("User:"),
    dcc.Dropdown(
        id='user-dropdown',
        options=[{'label': user, 'value': user} for user in ['Alle'] + sorted(df['user'].unique())],
        value='Alle'
    ),

    html.Label("CityMap:"),
    dcc.Dropdown(
        id='citymap-dropdown',
        options=[{'label': city_map, 'value': city_map} for city_map in ['Alle'] + sorted(df['CityMap'].unique())],
        value='Alle'
    ),

    html.Label("AOI-Typ:"),
    dcc.Dropdown(
        id='aoi-type-dropdown',
        options=[
            {'label': 'Polygonale AOI', 'value': 'Polygonale AOI'},
            {'label': 'Bounding Box AOI', 'value': 'Bounding Box AOI'}
        ],
        value='Polygonale AOI'
    ),

    dcc.Graph(id='aoi-plot')
])

# Callback zur Aktualisierung des AOI-Plots basierend auf den Dropdown-Auswahlen
@app.callback(
    Output('aoi-plot', 'figure'),
    [Input('user-dropdown', 'value'),
     Input('citymap-dropdown', 'value'),
     Input('aoi-type-dropdown', 'value')]
)
def update_aoi_plot(selected_user, selected_citymap, selected_aoi_type):
    filtered_df = df.copy()
    if selected_user != 'Alle':
        filtered_df = filtered_df[filtered_df['user'] == selected_user]
    if selected_citymap != 'Alle':
        filtered_df = filtered_df[filtered_df['CityMap'] == selected_citymap]

    if filtered_df.empty:
        return go.Figure()

    coords = filtered_df[['MappedFixationPointX', 'MappedFixationPointY']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=0).fit(coords)

    fig = go.Figure()

    # Farben für die AOIs
    colors = ['rgba(255, 0, 0, 0.3)', 'rgba(0, 255, 0, 0.3)', 'rgba(0, 0, 255, 0.3)']

    fig.add_trace(go.Scatter(
        x=coords['MappedFixationPointX'],
        y=coords['MappedFixationPointY'],
        mode='markers',
        marker=dict(color=kmeans.labels_, colorscale='Viridis'),
        name='Fixationspunkte'
    ))

    if selected_aoi_type == 'Polygonale AOI':
        for i in range(3):
            cluster_points = coords[kmeans.labels_ == i].values
            if len(cluster_points) >= 3:
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                hull_points = np.append(hull_points, [hull_points[0]], axis=0)  # Polygon schließen
                fig.add_trace(go.Scatter(
                    x=hull_points[:, 0],
                    y=hull_points[:, 1],
                    mode='lines',
                    fill='toself',
                    fillcolor=colors[i % len(colors)],
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'Cluster {i} AOI'
                ))

    elif selected_aoi_type == 'Bounding Box AOI':
        for i in range(3):
            cluster_points = coords[kmeans.labels_ == i]
            min_x, min_y = cluster_points.min()
            max_x, max_y = cluster_points.max()
            fig.add_shape(
                type='rect',
                x0=min_x, y0=min_y,
                x1=max_x, y1=max_y,
                line=dict(color=colors[i % len(colors)], dash='dash'),
                fillcolor=colors[i % len(colors)],
                opacity=0.3,
                name=f'Cluster {i} AOI'
            )

    fig.update_layout(
        title=f'AOI-Visualisierung ({selected_aoi_type})',
        xaxis_title='MappedFixationPointX',
        yaxis_title='MappedFixationPointY',
        yaxis=dict(autorange='reversed')
    )

    return fig

# Anwendung ausführen
if __name__ == '__main__':
    app.run_server(debug=True)
