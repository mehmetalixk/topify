import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Lambda
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from metrics import *
from data_encode import *


# Function to create pairs of input data with respect to one hot encoded label
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


# Create neural network
def create_base_network(input_size):
    model0 = Sequential()
    model0.add(Dense(units=128, input_shape=(input_size,), activation='relu'))
    model0.add(Dropout(rate=0.5))
    model0.add(Dense(units=128, activation='relu'))
    model0.add(Dropout(rate=0.5))
    model0.add(Dense(units=128, activation='relu'))
    model0.add(Dropout(rate=0.5))
    model0.add(Dense(units=128, activation='relu'))

    return model0


# Calculate distance
def euclidean_distance(vects):
    v1, v2 = vects
    return K.sqrt(K.sum(K.square(v1 - v2), axis=1, keepdims=True))


# Determine distance function's output shape
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


tracks = get_shuffled_data()
# Normalize input values
tracks = (tracks - tracks.min(axis=0)) / (tracks.max(axis=0) - tracks.min(axis=0))
tracks = tracks * (tracks.max() - tracks.min()) + tracks.min()

tracks = tracks.to_numpy()  # Turn pandas.dataframe dataset to numpy array
labels = np.copy(tracks[:, -1]).astype(int)  # Get last column of the data
tracks = np.delete(tracks, 9, 1)  # Delete last column of the data

X = tracks  # Dataset
y = labels  # Targets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)  # Split 10% of the data for test data

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)  # Split remaining 10% of the data for validation data

input_shape = X_train.shape[1:][0]  # Get feature count
num_classes = 2  # Number of classes

# Create one hot encoded labels
digit_indices_train = [np.where(y_train == i)[0] for i in range(num_classes)]
digit_indices_test = [np.where(y_test == i)[0] for i in range(num_classes)]
digit_indices_val = [np.where(y_val == i)[0] for i in range(num_classes)]

# Create input pairs
tr_pairs, tr_y = create_pairs(X_train, digit_indices_train)
te_pairs, te_y = create_pairs(X_test, digit_indices_test)
val_pairs, val_y = create_pairs(X_val, digit_indices_val)

base_network = create_base_network(input_shape)  # Create the model
# Create Inputs
input_a = Input(shape=(input_shape,))
input_b = Input(shape=(input_shape,))

# Put inputs into the model
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Lambda function to calculate the distance
distance = Lambda(function=euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = Model(inputs=[input_a, input_b], outputs=distance)

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

history = model.fit(x=[tr_pairs[:, 0], tr_pairs[:, 1]], y=tr_y, batch_size=256, epochs=200,
                    validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_y))

# Plot the graph of validation and training
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
plt.plot(training_loss, 'r--')
plt.plot(validation_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])  # Predict the test data
y_pred = y_pred.reshape(-1)  # Reshape it from 2D to 1D
y_pred[y_pred > 0] = 1.  # For our model we need to change every non zero number to one,

# Calculate true positive, false negative, false positive and true negative
tn, fp, fn, tp = confusion_matrix(te_y, y_pred).ravel()
print(f"TP:{tp}, FN:{fn}, FP:{fp}, TN:{tn}")

# Evaluate the test data
loss, accuracy, f1_score, precision, recall = model.evaluate([te_pairs[:, 0], te_pairs[:, 1]], te_y, verbose=0)
print('Loss on test set: %0.2f%%' % (100 * loss))
print('Accuracy on test set: %0.2f%%' % (100 * accuracy))
print('F1_score on test set: %0.2f%%' % (100 * f1_score))
print('Precision on test set: %0.2f%%' % (100 * precision))
print('Recall on test set: %0.2f%%' % (100 * recall))
