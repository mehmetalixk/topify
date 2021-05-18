from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
import pandas as pd

toptracks = pd.read_csv("unique_toptracks.csv")
toptracks["TopList"] = 1  # if the song is i a top list value is 1
normaltracks = pd.read_csv("separated_unique_tracks.csv")
normaltracks["TopList"] = 0
tracks = toptracks.append(normaltracks, ignore_index=True)

df = tracks.drop(['Artist name', 'Song name', 'Track URI', 'Key', "Mode", "Year", "Duration(ms)"], 1)
data = df.values

X, y = data[:, :-1], data[:, -1]

y = LabelEncoder().fit_transform(y)

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

balanced = pd.DataFrame(X, columns=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness',
                                    'Instrumentalness', 'Liveness', 'Valence', 'Tempo'])
balanced['TopList'] = y

counter = Counter(y)
for k, v in counter.items():
    per = v / len(y) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.savefig("oversample.png")

# balanced.to_csv("finaltracks.csv")
