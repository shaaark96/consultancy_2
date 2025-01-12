import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from dash import Dash, dcc, html, Input, Output

# Load the data
file_path = r'C:\Users\pelle\Documents\GitHub\consultancy_2\assets\all_fixation_data_cleaned_up.csv'
data = pd.read_csv(file_path, delimiter=';')

# Drop rows with missing values in relevant columns
data = data.dropna(subset=['MappedFixationPointX', 'MappedFixationPointY'])

# Extract fixation points for clustering
fixation_points = data[['MappedFixationPointX', 'MappedFixationPointY']]

# Apply KMeans clustering to generate AOIs
n_clusters = 3  # Adjust number of AOIs (clusters) as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['AOI'] = kmeans.fit_predict(fixation_points)

# Map AOIs to distinct colors
data['AOI'] = data['AOI'].astype(str)  # Convert AOI to string for better labeling in Plotly

# Initialize the Dash app
app = Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Interactive Scarf Plot of Eye Movements"),
    dcc.Dropdown(
        id='citymap-filter',
        options=[{'label': city, 'value': city} for city in data['CityMap'].unique()],
        value=data['CityMap'].unique()[0],  # Default value
        multi=False,
        placeholder="Select a CityMap",
        style={'width': '50%'}
    ),
    dcc.Dropdown(
        id='user-filter',
        options=[{'label': user, 'value': user} for user in data['user'].unique()],
        value=None,  # No default value, allows for all users
        multi=False,
        placeholder="Select a User",
        style={'width': '50%'}
    ),
    dcc.Graph(id='scarf-plot')
])


# Callback to update the plot
@app.callback(
    Output('scarf-plot', 'figure'),
    [Input('citymap-filter', 'value'),
     Input('user-filter', 'value')]
)
def update_plot(selected_citymap, selected_user):
    # Filter the data based on selections
    filtered_data = data
    if selected_citymap:
        filtered_data = filtered_data[filtered_data['CityMap'] == selected_citymap]
    if selected_user:
        filtered_data = filtered_data[filtered_data['user'] == selected_user]

    # Create the scatter plot
    fig = px.scatter(
        filtered_data,
        x='FixationDuration',
        y='user',
        color='AOI',
        title="Scarf Plot of Eye Movements (Interactive)",
        labels={
            'FixationIndex': 'Fixation Index (Sequence)',
            'user': 'Users',
            'AOI': 'AOIs (Clusters)',
        },
        size_max=10,  # Adjust size of points
        opacity=0.7
    )

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Fixation Index (Sequence)",
        yaxis_title="Users",
        legend_title="AOIs (Clusters)",
        yaxis=dict(categoryorder='category ascending')  # Ensure users are in ascending order
    )
    return fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
