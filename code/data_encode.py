from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def get_shuffled_data():
    top_tracks = pd.read_csv("../data/unique_toptracks.csv")
    top_tracks["TopList"] = 1  # if the song is i a top list value is 1

    normal_tracks = pd.read_csv("../data/separated_unique_tracks.csv")
    normal_tracks["TopList"] = 0

    tracks = top_tracks.append(normal_tracks, ignore_index=True)  # all tracks
    tracks = tracks.sample(frac=1)  # shuffle the data
    tracks.drop(["Artist name", "Song name", "Track URI", "Key", "Mode", "Year", "Duration(ms)"], axis=1,
                inplace=True)  # drop categorical features

    return tracks


def get_tracks():
    top_tracks = pd.read_csv("../data/unique_toptracks.csv")
    top_tracks["TopList"] = 1  # if the song is i a top list value is 1
    normal_tracks = pd.read_csv("../data/separated_unique_tracks.csv")
    normal_tracks["TopList"] = 0
    tracks = top_tracks.append(normal_tracks, ignore_index=True)

    # Dropping unneeded attributes of track
    df = tracks.drop(['Artist name', 'Song name', 'Track URI', 'Key', "Mode", "Year", "Duration(ms)"], 1)

    return df.values


def print_samples(samples):
    counter = Counter(samples)
    for k, v in counter.items():
        per = v / len(samples) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

    # plot the distribution
    pyplot.bar(counter.keys(), counter.values())
    pyplot.show()


data = get_tracks()
X, y = data[:, :-2], data[:, -2]
y = LabelEncoder().fit_transform(y)
print_samples(y)
