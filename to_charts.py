import pandas as pd
import plotly.graph_objects as go
import plotly as plt
import matplotlib.pyplot as plot
import numpy as np


def to_chart(songr, filename, num):
    print(songr)
    fig = go.Figure(data=go.Scatterpolar(r=songr,
                                         theta=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness',
                                                'Instrumentalness', 'Liveness', 'Valence', 'Tempo'],
                                         fill='toself'))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            ),
        ),
        showlegend=False)

    # fig.show()
    fig.write_image(filename + "/fig" + str(num) + ".jpeg")


toptracks = pd.read_csv("unique_toptracks.csv")
toptracks["TopList"] = 1  # if the song is i a top list value is 1
normaltracks = pd.read_csv("separated_unique_tracks.csv")
normaltracks["TopList"] = 0
tracks = toptracks.append(normaltracks, ignore_index=True)  # all tracks

tracks = tracks.sample(frac=1)  # shuffle the data
tracks.drop(["Artist name", "Song name", "Track URI", "Key", "Mode", "Year", "Duration(ms)"], axis=1,
            inplace=True)  # drop categorical features
toplist = tracks.iloc[:, [-1]]
print(tracks)
# tracks.drop(["TopList"])

column = tracks["Tempo"]
max_value = column.max()
# tracks["Tempo"]/=max_value
print(tracks)
tracksN = ((tracks - tracks.min()) / (tracks.max() - tracks.min()))
print(tracksN)
tracks = tracksN.to_numpy()

print(tracks)
index = 1
for track in tracks:
    if (track[-1] == 1):
        to_chart(track[:-1], "toptracks", index)
    else:
        to_chart(track[:-1], "normaltracks", index)
    index += 1
