import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Funktion zur Erstellung von Clustern basierend auf der Stadt-, CityMap- und Benutzerauswahl
def erstelle_cluster_nach_stadt_map_user(df, stadt=None, city_map=None, user=None, n_clusters=3):
    if stadt:
        stadt_data = df[df['City'] == stadt]
    else:
        stadt_data = df

    if city_map:
        stadt_data = stadt_data[stadt_data['CityMap'] == city_map]

    if user:
        stadt_data = stadt_data[stadt_data['user'] == user]

    # Überprüfen, ob Daten für die Analyse vorhanden sind
    if stadt_data.empty:
        print(f'Keine Daten für die Auswahl: Stadt={stadt}, CityMap={city_map}, Benutzer={user} vorhanden.')
        return

    # Clustering der Fixationspunkte
    coords = stadt_data[['MappedFixationPointX', 'MappedFixationPointY']].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)

    # Plot der AOI-Cluster
    plt.figure(figsize=(10, 7))
    plt.scatter(coords['MappedFixationPointX'], coords['MappedFixationPointY'], c=kmeans.labels_, cmap='viridis',
                marker='o')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x')
    plt.title(f'AOI-Cluster für Stadt={stadt}, CityMap={city_map}, Benutzer={user}')
    plt.xlabel('MappedFixationPointX')
    plt.ylabel('MappedFixationPointY')
    plt.gca().invert_yaxis()  # Y-Achse invertieren, um Pixelkoordinaten korrekt darzustellen
    plt.show()


# Laden der Daten
file_path = 'Antwerpen_s1_s2.xlsx'
df = pd.read_excel(file_path, sheet_name='Tabelle1')

# Beispielaufruf der Funktion
erstelle_cluster_nach_stadt_map_user(df, stadt='Antwerpen', city_map='Antwerpen_S1', user='p17')
erstelle_cluster_nach_stadt_map_user(df, stadt='Antwerpen', city_map='Antwerpen_S2', user='p4')
