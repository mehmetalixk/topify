import spotipy
import os
from spotipy.oauth2 import SpotifyOAuth
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

client_id = os.getenv('SPOTIPY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')

scope = "user-library-read"
judas = 'spotify:artist:2tRsMl4eGxwoNabM08Dm4I'
spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                    client_secret=client_secret,
                                                    redirect_uri=redirect_uri, scope=scope))

results = spotify.artist_albums(judas, album_type='album')
albums = results['items']

while results['next']:
    results = spotify.next(results)
    albums.extend(results['items'])

for album in albums:
    img = Image.open(requests.get(album['images'][0]['url'], stream=True).raw)
    img.show()
