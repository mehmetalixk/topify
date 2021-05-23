from imblearn.over_sampling import SMOTE
from data_encode import *

data = get_tracks()
X, y = data[:, :-1], data[:, -1]
y = LabelEncoder().fit_transform(y)

# Oversampling tracks
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

balanced = pd.DataFrame(X, columns=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness',
                                    'Instrumentalness', 'Liveness', 'Valence', 'Tempo'])
balanced['TopList'] = y
print_samples(y)

"""
To save oversampling;
balanced.to_csv("final_tracks.csv")
"""