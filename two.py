import os

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
import pickle
from dotenv import load_dotenv

load_dotenv()
song_cluster_pipeline = pickle.load(open("mrs_model", 'rb'))

data = pd.read_csv("data.csv")



SPOTIFY_CLIENT_ID = "d5a1ba2ad8334bb4a5465f8a86d33ef6"
'''
SPOTIFY_CLIENT_SECRET = "97668849339d449582d3d30feb653a65"
'''


SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')


'''
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
'''

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID,
                                                           client_secret=SPOTIFY_CLIENT_SECRET))


def find_song(name, year):    #(name, year)
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    #results = sp.search(q= 'track: {}'.format(name), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    #song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                 & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])


def get_mean_vector(song_list, spotify_data):
    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict


def find_song_year(name):    #(name, year)
    result = sp.search(name)
    if result['tracks']['items'] == []:
        return None
    result = result['tracks']['items'][0]
    album = sp.album(result["album"]["external_urls"]["spotify"])
    #print("album release-date:", album["release_date"])
    datef = album["release_date"]
    date = datef.split("-")[0]
    return int(date)

def recommend_songs(song_list, spotify_data = data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    #print(song_list)
    song_dict = flatten_dict_list(song_list)
    #print(song_dict)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

#print(recommend_songs([{"name":"Beat It","year": 1982}]))

def recommendations(song_name):
    song = ' '.join(elem.capitalize() for elem in song_name.split())
    songList = [{"name": song, "year": find_song_year(song)}]
    return recommend_songs(songList)
