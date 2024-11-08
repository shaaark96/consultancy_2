import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import seaborn as sns

# Funktion zur Erstellung von AOIs basierend auf Clustern
def erstelle_heatmap_aoi(df, stadt=None, city_map=None, user=None):
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

    plt.figure(figsize=(10, 7))
    sns.kdeplot(x=coords['MappedFixationPointX'], y=coords['MappedFixationPointY'], cmap='Reds', fill=True)
    plt.title(f'Heatmap AOIs für Stadt={stadt}, CityMap={city_map}, Benutzer={user}')
    plt.xlabel('MappedFixationPointX')
    plt.ylabel('MappedFixationPointY')
    plt.gca().invert_yaxis()
    plt.show()

# Laden der Daten
file_path = 'Antwerpen_s1_s2.xlsx'
df = pd.read_excel(file_path, sheet_name='Tabelle1')

# Beispielaufruf der Funktion
erstelle_heatmap_aoi(df, stadt='Antwerpen', city_map='Antwerpen_S1', user='p17')
erstelle_heatmap_aoi(df, stadt='Antwerpen', city_map='Antwerpen_S2', user='p4')
