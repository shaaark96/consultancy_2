import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import numpy as np
from PIL import Image
import glob
import plotly.express as px

# Daten laden und vorbereiten
data_path = '../assets/all_fixation_data_cleaned_up.csv'  # Korrigieren Sie den Dateipfad
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

# Funktion zur Bildpfadermittlung für color
def get_image_path_color(selected_city):
    file_pattern_color = f'assets/*_{selected_city}_Color.jpg'
    matching_files = glob.glob(file_pattern_color)
    if matching_files:
        image_path = matching_files[0]
        img = Image.open(image_path)
        width, height = img.size
        return image_path, width, height
    return None, None, None

# Funktion zur Bildpfadermittlung für grey
def get_image_path_grey(selected_city):
    file_pattern_grey = f'assets/*_{selected_city}_Grey.jpg'
    matching_files = glob.glob(file_pattern_grey)
    if matching_files:
        image_path = matching_files[0]
        img = Image.open(image_path)
        width, height = img.size
        return image_path, width, height
    return None, None, None

# Funktion zur Normalisierung und Filterung von Daten
def normalize_and_filter_data(df, selected_city, width, height):
    if selected_city == 'Antwerpen_S1':
        df['NormalizedPointX'] = (df['MappedFixationPointX'] / 1651.00) * width
        df['NormalizedPointY'] = (df['MappedFixationPointY'] / 1200.00) * height
    else:
        df['NormalizedPointX'] = df['MappedFixationPointX']
        df['NormalizedPointY'] = df['MappedFixationPointY']

    # Filtert die Daten innerhalb der Bildgrenzen
    df = df[(df['NormalizedPointX'] >= 0) & (df['NormalizedPointX'] <= width) &
            (df['NormalizedPointY'] >= 0) & (df['NormalizedPointY'] <= height)]
    return df

# Layout der Anwendung
app.layout = html.Div([
    html.H1("AOI-Visualisierung mit Clusteranalyse für Color und Grey"),

    # Dropdown für die Auswahl der Stadt
    html.Div([
        html.Label("City:"),
        dcc.Dropdown(
            id='city_dropdown',
            options=[{'label': city, 'value': city} for city in ['Alle'] + sorted(df['CityMap'].unique())],
            value='Alle'
        ),
    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    # Dropdowns und Eingabefelder für Benutzer und AOI-Typ in "color" und "grey"
    html.Div([
        html.Label("User (Color):"),
        dcc.Dropdown(id='dropdown_user_color'),

        html.Label("Anzahl der Cluster (Color):"),
        dcc.Input(id='n_clusters_color', type='number', value=3, min=1, step=1),

        html.Label("AOI-Typ (Color):"),
        dcc.Dropdown(
            id='aoi_type_dropdown_color',
            options=[
                {'label': 'Polygonale AOI', 'value': 'Polygonale AOI'},
                {'label': 'Bounding Box AOI', 'value': 'Bounding Box AOI'}
            ],
            value='Polygonale AOI'
        ),

        html.Label("User (Grey):"),
        dcc.Dropdown(id='dropdown_user_grey'),

        html.Label("Anzahl der Cluster (Grey):"),
        dcc.Input(id='n_clusters_grey', type='number', value=3, min=1, step=1),

        html.Label("AOI-Typ (Grey):"),
        dcc.Dropdown(
            id='aoi_type_dropdown_grey',
            options=[
                {'label': 'Polygonale AOI', 'value': 'Polygonale AOI'},
                {'label': 'Bounding Box AOI', 'value': 'Bounding Box AOI'}
            ],
            value='Polygonale AOI'
        )
    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    # Layout für Plots
    html.Div([
        html.H3("Color Plot"),
        dcc.Graph(id='aoi_plot_color'),
    ], style={'width': '45%', 'display': 'inline-block'}),

    html.Div([
        html.H3("Grey Plot"),
        dcc.Graph(id='aoi_plot_grey'),
    ], style={'width': '45%', 'display': 'inline-block'})
])

# Callback zur Aktualisierung der Benutzer-Dropdowns basierend auf der Auswahl der Stadt
@app.callback(
    [Output('dropdown_user_color', 'options'),
     Output('dropdown_user_grey', 'options')],
    [Input('city_dropdown', 'value')]
)
def update_user_dropdowns(selected_city):
    if selected_city and selected_city != 'Alle':
        color_users = df[(df['CityMap'] == selected_city) & (df['description'] == 'color')]['user'].unique()
        grey_users = df[(df['CityMap'] == selected_city) & (df['description'] == 'grey')]['user'].unique()
    else:
        color_users = df[df['description'] == 'color']['user'].unique()
        grey_users = df[df['description'] == 'grey']['user'].unique()

    color_options = [{'label': user, 'value': user} for user in sorted(color_users)]
    grey_options = [{'label': user, 'value': user} for user in sorted(grey_users)]

    return color_options, grey_options

# Callback zur Aktualisierung des Color-Plots und Bildes
@app.callback(
    Output('aoi_plot_color', 'figure'),
    [Input('dropdown_user_color', 'value'),
     Input('city_dropdown', 'value'),
     Input('aoi_type_dropdown_color', 'value'),
     Input('n_clusters_color', 'value')]
)
def update_color_plot(selected_user, selected_city, selected_aoi_type, n_clusters):
    filtered_df = df[df['description'] == 'color']
    if selected_user and selected_user != 'Alle':
        filtered_df = filtered_df[filtered_df['user'] == selected_user]
    if selected_city and selected_city != 'Alle':
        filtered_df = filtered_df[filtered_df['CityMap'] == selected_city]

    if filtered_df.empty:
        return go.Figure()

    # Bildpfad für color ermitteln und Daten normalisieren
    image_path_color, width, height = get_image_path_color(selected_city)
    if image_path_color and width and height:
        filtered_df = normalize_and_filter_data(filtered_df.copy(), selected_city, width, height)

    # Prüfen Sie, ob die Spalten vorhanden sind, bevor der Plot erstellt wird
    if 'NormalizedPointX' not in filtered_df.columns or 'NormalizedPointY' not in filtered_df.columns:
        print("Fehler: NormalizedPointX oder NormalizedPointY fehlt im DataFrame.")
        return go.Figure()

    # Erstellen des Color-Plots
    color_fig = create_aoi_plot(filtered_df[['NormalizedPointX', 'NormalizedPointY']], n_clusters, selected_aoi_type,'Color', image_path_color, width, height)

    return color_fig

# Callback zur Aktualisierung des Grey-Plots und Bildes
@app.callback(
    Output('aoi_plot_grey', 'figure'),
    [Input('dropdown_user_grey', 'value'),
     Input('city_dropdown', 'value'),
     Input('aoi_type_dropdown_grey', 'value'),
     Input('n_clusters_grey', 'value')]
)
def update_grey_plot(selected_user, selected_city, selected_aoi_type, n_clusters):
    print(f"Selected user: {selected_user}, Selected city: {selected_city}, AOI type: {selected_aoi_type}, Clusters: {n_clusters}")

    filtered_df = df[df['description'] == 'grey']
    if selected_user and selected_user != 'Alle':
        filtered_df = filtered_df[filtered_df['user'] == selected_user]
    if selected_city and selected_city != 'Alle':
        filtered_df = filtered_df[filtered_df['CityMap'] == selected_city]

    if filtered_df.empty:
        print("Filtered DataFrame is empty.")
        return go.Figure()

    image_path_grey, width, height = get_image_path_grey(selected_city)
    if not image_path_grey or not width or not height:
        print("Error: Image path or dimensions not found.")
        return go.Figure()

    filtered_df = normalize_and_filter_data(filtered_df.copy(), selected_city, width, height)
    print("Filtered DataFrame (after normalization):", filtered_df.head())

    if 'NormalizedPointX' not in filtered_df.columns or 'NormalizedPointY' not in filtered_df.columns:
        print("Error: Normalized columns are missing.")
        return go.Figure()

    return create_aoi_plot(filtered_df[['NormalizedPointX', 'NormalizedPointY']], n_clusters, selected_aoi_type, 'Grey', image_path_grey, width, height)


# Funktion zur Erstellung des AOI-Plots
def create_aoi_plot(coords, n_clusters, aoi_type, plot_title, image_path=None, width=None, height=None):
    if coords.empty:
        return go.Figure()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    fig = go.Figure()

    # Bild als Hintergrund hinzufügen
    fig.add_layout_image(
        dict(
            source=image_path,
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

    # Farben für die AOIs
    colors = ['rgba(255, 0, 0, 0.3)', 'rgba(0, 255, 0, 0.3)', 'rgba(0, 0, 255, 0.3)', 'rgba(255, 255, 0, 0.3)', 'rgba(255, 0, 255, 0.3)']

    # Scatter-Plot für die Fixationspunkte
    fig.add_trace(go.Scatter(
        x=coords['NormalizedPointX'],
        y=coords['NormalizedPointY'],
        mode='markers',
        marker=dict(color=kmeans.labels_, colorscale='Viridis', opacity=0.6),
        name='Fixationspunkte'
    ))

    # AOI-Zeichnung basierend auf dem ausgewählten Typ
    if aoi_type == 'Polygonale AOI':
        for i in range(n_clusters):
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

    elif aoi_type == 'Bounding Box AOI':
        for i in range(n_clusters):
            cluster_points = coords[kmeans.labels_ == i]
            min_x, min_y = cluster_points.min()
            max_x, max_y = cluster_points.max()
            fig.add_shape(
                type='rect',
                x0=min_x, y0=min_y,
                x1=max_x, y1=max_y,
                line=dict(color=colors[i % len(colors)], dash='dash'),
                fillcolor=colors[i % len(colors)],
                opacity=0.3
            )

    fig.update_layout(
        title=f'AOI-Visualisierung für {plot_title} ({aoi_type}, {n_clusters} Cluster)',
        xaxis=dict(title='MappedFixationPointX', range=[0, width], showgrid=False),
        yaxis=dict(title='MappedFixationPointY', range=[0, height], showgrid=False, scaleanchor="x", scaleratio=1),
        yaxis_autorange='reversed',  # Y-Achse umkehren, damit das Bild von oben nach unten korrekt angezeigt wird
        images=[{
            'xref': 'x',
            'yref': 'y',
            'x': 0,
            'y': 0,
            'sizex': width,
            'sizey': height,
            'layer': 'below',
            'source': image_path
        }]
    )
    return fig

# Anwendung ausführen
if __name__ == '__main__':
    app.run_server(debug=True)