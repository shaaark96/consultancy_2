import plotly.express as px
import pandas as pd

# Beispiel-Daten in ein DataFrame umwandeln
data = [
    {"User": "User 1", "Start": 0, "End": 2, "AOI": "AOI 1"},
    {"User": "User 1", "Start": 2, "End": 4, "AOI": "AOI 2"},
    {"User": "User 1", "Start": 4, "End": 6, "AOI": "AOI 3"},
    {"User": "User 2", "Start": 0, "End": 3, "AOI": "AOI 2"},
    {"User": "User 2", "Start": 3, "End": 5, "AOI": "AOI 1"},
    {"User": "User 2", "Start": 5, "End": 6, "AOI": "AOI 3"},
    {"User": "User 3", "Start": 0, "End": 1, "AOI": "AOI 3"},
    {"User": "User 3", "Start": 1, "End": 4, "AOI": "AOI 1"},
    {"User": "User 3", "Start": 4, "End": 6, "AOI": "AOI 2"},
]

# DataFrame erstellen
df = pd.DataFrame(data)

# Dauer der Fixation berechnen
df["Duration"] = df["End"] - df["Start"]

# Scarfplot mit Plotly Express erstellen
fig = px.bar(
    df,
    x="Start",  # Zeit (Startzeit der Fixation)
    y="User",  # User auf der Y-Achse
    color="AOI",  # Farbcodierung nach AOI
    text="AOI",  # AOI-Label optional
    orientation="h",  # Horizontale Balken
    width=900,  # Plot-Breite
    height=500,  # Plot-Höhe
)

# Balkenbreite (Fixationsdauer) anpassen
fig.update_traces(
    base=0,
    width=df["Duration"],  # Die Dauer als Breite der Balken verwenden
    textposition="inside",  # Text innerhalb der Balken
)

# Achsentitel und Layout
fig.update_layout(
    title="Scarfplot: Fixationen auf AOIs über die Zeit",
    xaxis_title="Zeit (s)",
    yaxis_title="User",
    legend_title="AOIs",
)

# Plot anzeigen
fig.show()
