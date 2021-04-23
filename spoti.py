import spotipy
import os
from spotipy.oauth2 import SpotifyOAuth
import csv

client_id = os.getenv('SPOTIPY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')

scope = "user-library-read app-remote-control user-modify-playback-state"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri, scope=scope))

playlist_items = sp.playlist_items("6yPiKpy7evrwvZodByKvM9")


def get_playlist_tracks(playlist_id):
    results = sp.playlist_items(playlist_id)
    tracks_ = results['items']
    while results['next']:
        results = sp.next(results)
        tracks_.extend(results['items'])
    return tracks_


def csv_creator(file):
    tracks_writer = csv.writer(file)
    tracks_writer.writerow(['Artist name', 'Song name', 'Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
                            'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
                            'Valence', 'Tempo', 'Duration(ms)', 'Track URI', 'TopList', 'Year'])
    return tracks_writer


def csv_write(playlist, csv_file, top_list='no'):
    for item in playlist:
        try:
            artist_name = item['track']['artists'][0]['name']  # Get artist name
            year = item['track']['album']['release_date'].split("-")[0]
            track_name = item['track']['name']  # Get track name
            track_uri = item['track']['uri']  # Get track URI
            features = sp.audio_features(track_uri)  # Extract features given track URI
            csv_file.writerow([artist_name, track_name, features[0]['danceability'], features[0]['energy'],
                               features[0]['key'], features[0]['loudness'], features[0]['mode'],
                               features[0]['speechiness'], features[0]['acousticness'], features[0]['instrumentalness'],
                               features[0]['liveness'], features[0]['valence'], features[0]['tempo'],
                               features[0]['duration_ms'], track_uri, top_list, year])
        except:
            continue


def remove_dup(file_path):
    with open(file_path, 'r') as in_file, open('unique_' + file_path, 'w') as out_file:
        seen = set()
        for line in in_file:
            if line in seen:
                continue
            seen.add(line)
            out_file.write(line)


def separation(toptracks_path, tracks_path):
    with open(toptracks_path, 'r') as toptracks:
        toplist = set()
        for line in toptracks:
            if line in toplist:
                continue
            if line.split(",")[1] == "Song name":
                continue
            toplist.add(line.split(",")[1])
    with open(tracks_path, 'r') as tracks, open('separated_'+tracks_path, 'w') as out_file:
        for line in tracks:
            if line.split(",")[1] in toplist:
                print(line)
                continue
            out_file.write(line)


remove_dup("toptracks.csv")
