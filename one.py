import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle



data = pd.read_csv(r"C:\Users\apollo\PycharmProjects\Flask Prajects Prototypes\Music Recommendation System\data.csv")
genre_data = pd.read_csv(r"C:\Users\apollo\PycharmProjects\Flask Prajects Prototypes\Music Recommendation System\data_by_genres.csv")
year_data = pd.read_csv(r"C:\Users\apollo\PycharmProjects\Flask Prajects Prototypes\Music Recommendation System\data_by_year.csv")

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20,verbose=False)) ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
pickle.dump(song_cluster_pipeline, open("mrs_model", 'wb'))