import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np

# Funktion zur Erstellung von AOIs basierend auf Clustern
def erstelle_polygonale_aoi(df, stadt=None, city_map=None, user=None, n_clusters=3):
    if stadt:
        stadt_data = df[df['City'] == stadt]
    else:
        stadt_data = df

    if city_map:
        stadt_data = stadt_data[stadt_data['CityMap'] == city_map]

    if user:
        stadt_data = stadt_data[stadt_data['user'] == user]

    if stadt_data.empty:
        print(f'Keine Daten für die Auswahl: Stadt={stadt}, CityMap={city_map}, Benutzer={user} vorhanden.')
        return

    coords = stadt_data[['MappedFixationPointX', 'MappedFixationPointY']].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)

    plt.figure(figsize=(10, 7))
    plt.scatter(coords['MappedFixationPointX'], coords['MappedFixationPointY'], c=kmeans.labels_, cmap='viridis',
                marker='o')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x')

    for i in range(n_clusters):
        cluster_points = coords[kmeans.labels_ == i].values
        if len(cluster_points) >= 3:  # Ein Polygon benötigt mindestens 3 Punkte
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'b--')

    plt.title(f'Polygonale AOIs für Stadt={stadt}, CityMap={city_map}, Benutzer={user}')
    plt.xlabel('MappedFixationPointX')
    plt.ylabel('MappedFixationPointY')
    plt.gca().invert_yaxis()
    plt.show()

# Laden der Daten
file_path = 'Antwerpen_s1_s2.xlsx'
df = pd.read_excel(file_path, sheet_name='Tabelle1')

# Beispielaufruf der Funktion
erstelle_polygonale_aoi(df, stadt='Antwerpen', city_map='Antwerpen_S1', user='p17', n_clusters=3)
erstelle_polygonale_aoi(df, stadt='Antwerpen', city_map='Antwerpen_S2', user='p4', n_clusters=3)
