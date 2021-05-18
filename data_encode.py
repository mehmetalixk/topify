from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
import pandas as pd


toptracks = pd.read_csv("unique_toptracks.csv")
toptracks["TopList"] = 1  # if the song is i a top list value is 1
normaltracks = pd.read_csv("separated_unique_tracks.csv")
normaltracks["TopList"] = 0
tracks = toptracks.append(normaltracks, ignore_index=True)

data = tracks.values

X, y = data[:, :-2], data[:, -2]
y = LabelEncoder().fit_transform(y)

counter = Counter(y)
for k, v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.savefig("without_balancing.png")
