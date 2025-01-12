import numpy as np
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def generate_cluster_colors(n_clusters):
    """
    Generates a consistent color mapping for clusters.

    Parameters:
        n_clusters (int): Number of clusters.

    Returns:
        dict: A dictionary mapping cluster index to rgba color strings.
    """
    base_colors = px.colors.qualitative.Plotly
    return {i: f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.8)'
            for i, color in enumerate(base_colors[:n_clusters])}


def add_aoi_visualization_grey(fig, coords, labels, selected_n_clusters_grey, selected_aoi_type_grey):
    """
    Adds AOI (Area of Interest) visualization to a Plotly figure for 'grey' visualization.

    Parameters:
        fig (go.Figure): The Plotly figure to add AOI visualizations.
        coords (pd.DataFrame): DataFrame containing the coordinates of points (columns: ['x', 'y']).
        labels (pd.Series): Cluster labels for each point in `coords`.
        selected_n_clusters_grey (int): Number of clusters to visualize.
        selected_aoi_type_grey (str): Type of AOI visualization ('Polygonale AOI' or 'Bounding Box AOI').

    Returns:
        go.Figure: The updated Plotly figure with AOI visualizations.
    """
    cluster_colors = generate_cluster_colors(selected_n_clusters_grey)

    if selected_aoi_type_grey == 'Polygonale AOI':
        for i in range(selected_n_clusters_grey):
            cluster_points = coords[labels == i].values
            if len(cluster_points) >= 3:
                try:
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    hull_points = np.append(hull_points, [hull_points[0]], axis=0)  # Close the polygon
                    fig.add_trace(go.Scatter(
                        x=hull_points[:, 0],
                        y=hull_points[:, 1],
                        mode='lines',
                        fill='toself',
                        fillcolor=cluster_colors[i],
                        line=dict(color='rgba(0,0,0,0)'),
                        name=f'Cluster {i} AOI'
                    ))
                except Exception as e:
                    print(f"Error creating ConvexHull for cluster {i}: {e}")
    elif selected_aoi_type_grey == 'Bounding Box AOI':
        for i in range(selected_n_clusters_grey):
            cluster_points = coords[labels == i]
            if not cluster_points.empty:
                min_x, min_y = cluster_points.min()
                max_x, max_y = cluster_points.max()
                fig.add_shape(
                    type='rect',
                    x0=min_x, y0=min_y,
                    x1=max_x, y1=max_y,
                    line=dict(color=cluster_colors[i], dash='dash'),
                    fillcolor=cluster_colors[i],
                    opacity=0.3
                )
    return fig


def create_scarf_plot(filtered_df, cluster_colors):
    """    Example function to create a scarf plot (implementation may vary based on requirements).

    Parameters:
        filtered_df (pd.DataFrame): DataFrame containing scarf plot data.
        cluster_colors (dict): Color mapping for clusters.

    Returns:
        go.Figure: The generated scarf plot figure.
    """
    fig = go.Figure()
    for cluster, color in cluster_colors.items():
        cluster_data = filtered_df[filtered_df['cluster'] == cluster]
        fig.add_trace(go.Scatter(
            x=cluster_data['x'],
            y=cluster_data['y'],
            mode='markers',
            marker=dict(color=color),
            name=f'Cluster {cluster}'
        ))
    return fig


# Example usage
# Mock Data
coords = pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100)})
labels = pd.Series(np.random.randint(0, 5, size=100))  # Mock labels for 5 clusters
filtered_df = pd.DataFrame(
    {'x': np.random.rand(100), 'y': np.random.rand(100), 'cluster': np.random.randint(0, 5, size=100)})
selected_n_clusters_grey = 5
selected_aoi_type_grey = 'Polygonale AOI'

# Initialize Plotly Figure
fig = go.Figure()

# Add AOI Visualization
fig = add_aoi_visualization_grey(fig, coords, labels, selected_n_clusters_grey, selected_aoi_type_grey)

# Generate Colors for Scarf Plot
cluster_grey = generate_cluster_colors(selected_n_clusters_grey)

# Create Scarf Plot
fig_scarf = create_scarf_plot(filtered_df, cluster_grey)

# Show Figures
fig.show()
fig_scarf.show()