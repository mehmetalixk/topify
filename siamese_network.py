import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Lambda
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def recall_m(y_true, y_predict):
    true_positives = K.sum(K.round(K.clip(y_true * y_predict, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_val = true_positives / (possible_positives + K.epsilon())
    return recall_val


def precision_m(y_true, y_predict):
    true_positives = K.sum(K.round(K.clip(y_true * y_predict, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_predict, 0, 1)))
    precision_val = true_positives / (predicted_positives + K.epsilon())
    return precision_val


def f1_m(y_true, y_predict):
    precision_val = precision_m(y_true, y_predict)
    recall_val = recall_m(y_true, y_predict)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))


def create_pairs(x, digit_indices):
    pairs = []
    target_labels = []

    n = min([len(digit_indices[d]) for d in range(2)]) - 1
    for d in range(num_classes):
        for i in range(n):
            p1, p2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[p1], x[p2]]]

            inc = random.randrange(1, 2)

            dn = (d + inc) // 2
            n1, n2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[n1], x[n2]]]
            target_labels += [1, 0]
    return np.array(pairs).astype("float32"), np.array(target_labels).astype("float32")


def create_base_network(input_size):
    model0 = Sequential()
    model0.add(Dense(units=128, input_shape=(input_size,), activation='relu'))
    model0.add(Dropout(rate=0.5))
    model0.add(Dense(units=128, activation='relu'))
    model0.add(Dropout(rate=0.5))
    model0.add(Dense(units=128, activation='relu'))

    return model0


def euclidean_distance(vects):
    v1, v2 = vects
    return K.sqrt(K.sum(K.square(v1 - v2), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


""""
toptracks = pd.read_csv("unique_toptracks.csv")
toptracks["TopList"] = 1
normaltracks = pd.read_csv("separated_unique_tracks.csv")
normaltracks["TopList"] = 0
tracks = toptracks.append(normaltracks, ignore_index=True)
tracks.drop(["Artist name", "Song name", "Track URI", "Key", "Mode", "Year", "Duration(ms)"], axis=1,
            inplace=True)
"""""

tracks = pd.read_csv("finaltracks.csv")
tracks = tracks.sample(frac=1)
tracks = (tracks - tracks.min(axis=0)) / (tracks.max(axis=0) - tracks.min(axis=0))
tracks = tracks * (tracks.max() - tracks.min()) + tracks.min()
tracks = tracks.to_numpy()
labels = np.copy(tracks[:, -1]).astype(int)
tracks = np.delete(tracks, 9, 1)

X = tracks
y = labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)

input_shape = X_train.shape[1:][0]
print("Input_Shape: ", input_shape)
num_classes = 2

digit_indices_train = [np.where(y_train == i)[0] for i in range(num_classes)]
digit_indices_test = [np.where(y_test == i)[0] for i in range(num_classes)]
digit_indices_val = [np.where(y_val == i)[0] for i in range(num_classes)]

tr_pairs, tr_y = create_pairs(X_train, digit_indices_train)
te_pairs, te_y = create_pairs(X_test, digit_indices_test)
val_pairs, val_y = create_pairs(X_val, digit_indices_val)

print("trainSet shape: ", tr_pairs.shape)
print("trainLabel shape: ", tr_y.shape)
print("testSet shape: ", te_pairs.shape)
print("testLabel shape: ", te_y.shape)
print("validationSet shape: ", val_pairs.shape)
print("validationLabel shape: ", val_y.shape)

base_network = create_base_network(input_shape)
input_a = Input(shape=(input_shape,))
input_b = Input(shape=(input_shape,))

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(function=euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = Model(inputs=[input_a, input_b], outputs=distance)
model.summary()

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

history = model.fit(x=[tr_pairs[:, 0], tr_pairs[:, 1]], y=tr_y, batch_size=32, epochs=600,
                    validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_y))

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

plt.plot(training_loss, 'r--')
plt.plot(validation_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])

print(confusion_matrix(te_y.reshape(te_y.shape[0], 1), y_pred.round()))

loss, accuracy, f1_score, precision, recall = model.evaluate([te_pairs[:, 0], te_pairs[:, 1]], te_y, verbose=0)
print('Loss on test set: %0.2f%%' % (100 * loss))
print('Accuracy on test set: %0.2f%%' % (100 * accuracy))
print('F1_score on test set: %0.2f%%' % (100 * f1_score))
print('Precision on test set: %0.2f%%' % (100 * precision))
print('Recall on test set: %0.2f%%' % (100 * recall))
