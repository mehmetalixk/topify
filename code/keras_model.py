import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from metrics import *
from data_encode import *


tracks = get_shuffled_data()

"""
Use it with SMOTE
tracks = pd.read_csv("../data/finaltracks.csv")
tracks = tracks.sample(frac=1)
"""

unscaled_inputs = tracks.iloc[:, 1:-1]
toplist = tracks.iloc[:, [-1]]
tracks = unscaled_inputs
tracks = (tracks - tracks.min(axis=0)) / (tracks.max(axis=0) - tracks.min(axis=0))
scaled_inputs = tracks * (tracks.max() - tracks.min()) + tracks.min()


samples_count = scaled_inputs.shape[0]
# split data into test,validation and train data
train_samples_count = int(0.8 * samples_count)  # 80%
validation_samples_count = int(0.1 * samples_count)  # 10%
test_samples_count = samples_count - train_samples_count - validation_samples_count  # 10%

# train data
train_inputs = scaled_inputs[:train_samples_count].astype(np.float)
train_targets = toplist[:train_samples_count].astype(np.int)
# validation data
validation_inputs = scaled_inputs[train_samples_count:train_samples_count + validation_samples_count].astype(np.float)
validation_targets = toplist[train_samples_count:train_samples_count + validation_samples_count].astype(np.int)
# test data
test_inputs = scaled_inputs[train_samples_count + validation_samples_count:].astype(np.float)
test_targets = toplist[train_samples_count + validation_samples_count:].astype(np.int)


# Setting the input and output sizes
input_size = 9  # count of features
output_size = 1  # count of targets
# Same hidden layer size for both hidden layers
hidden_layer_size = 128  # counts of neurons

# the model
model = tf.keras.Sequential([
    # tf.keras.layers.Dense is to output = activation(dot(input, weight) + bias)
    # hidden_layer_size and the activation function is important
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),  # 1st hidden layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),  # 2nd hidden layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),  # 3rd hidden layer
    # activate it with sigmoid
    tf.keras.layers.Dense(output_size, activation='sigmoid')  # output layer
])

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

# batch size
batch_size = 128
# maximum number of training epochs
max_epochs = 600

# fit the model
# the train, validation and test data are not iterable
history = model.fit(train_inputs,  # train inputs
                    train_targets,  # train targets
                    batch_size=batch_size,  # batch size
                    epochs=max_epochs,  # epochs that we will train for (assuming early stopping doesn't kick in)
                    # callbacks are functions called by a task when a task is completed
                    # task here is to check if val_loss is increasing
                    # callbacks=[early_stopping], # early stopping
                    validation_data=(validation_inputs, validation_targets),  # validation data
                    verbose=2,  # making sure we get enough information about the training process
                    )
# Get training and test loss histories
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, validation_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

loss, accuracy, f1_score, precision, recall = model.evaluate(test_inputs, test_targets, verbose=0)
y_pred = model.predict(test_inputs, batch_size=batch_size)

y_pred = y_pred.reshape(-1)
y_pred[y_pred <= 0.5] = 0.
y_pred[y_pred > 0.5] = 1.
tn, fp, fn, tp = confusion_matrix(test_targets, y_pred).ravel()
print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
