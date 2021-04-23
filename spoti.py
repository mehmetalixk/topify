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

playlist_items = sp.playlist_items("37i9dQZF1DWSV3Tk4GO2fq")

tracks = open('tracks.csv', 'a')

tracks_writer = csv.writer(tracks)
"""tracks_writer.writerow(['Artist name', 'Song name', 'Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
                        'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
                        'Valence', 'Tempo', 'Duration(ms)', 'Track URI', 'TopList', 'Year'])"""

for item in playlist_items['items']:
    artist_name = item['track']['artists'][0]['name']  # Get artist name
    track_name = item['track']['name']  # Get track name
    track_uri = item['track']['uri']  # Get track URI
    features = sp.audio_features(track_uri)  # Extract features given track URI

    tracks_writer.writerow([artist_name, track_name, features[0]['danceability'], features[0]['energy'],
                            features[0]['key'], features[0]['loudness'], features[0]['mode'],
                            features[0]['speechiness'], features[0]['acousticness'], features[0]['instrumentalness'],
                            features[0]['liveness'], features[0]['valence'], features[0]['tempo'],
                            features[0]['duration_ms'], track_uri, 'yes', '1950'])
