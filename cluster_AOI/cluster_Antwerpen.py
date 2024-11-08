import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Laden der Daten
file_path = 'Antwerpen.xlsx'
df = pd.read_excel(file_path, sheet_name='Tabelle2')


# Filtern der Daten für eine bestimmte Stadt
def erstelle_cluster(df, stadt, user=None, n_clusters=3):
    stadt_data = df[df['City'] == stadt]

    # Nach Benutzer filtern, falls angegeben
    if user:
        user_data = stadt_data[stadt_data['user'] == user]
    else:
        user_data = stadt_data

    # Überprüfen, ob Daten für die Analyse vorhanden sind
    if user_data.empty:
        print(f'Keine Daten für Benutzer {user} in der Stadt {stadt} vorhanden.')
        return

    # Clustering der Fixationspunkte
    coords = user_data[['MappedFixationPointX', 'MappedFixationPointY']].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)

    # Plot der AOI-Cluster
    plt.figure(figsize=(10, 7))
    plt.scatter(coords['MappedFixationPointX'], coords['MappedFixationPointY'], c=kmeans.labels_, cmap='viridis',
                marker='o')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x')
    plt.title(f'AOI-Cluster für {"alle Benutzer" if not user else "Benutzer " + user} in {stadt}')
    plt.xlabel('MappedFixationPointX')
    plt.ylabel('MappedFixationPointY')
    plt.gca().invert_yaxis()  # Y-Achse invertieren, um Pixelkoordinaten korrekt darzustellen
    plt.show()


# Beispielaufruf der Funktion
erstelle_cluster(df, 'Antwerpen')
erstelle_cluster(df, 'Antwerpen', user='p15')

