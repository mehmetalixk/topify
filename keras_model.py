import numpy as np # linear algebra
import pandas as pd # data processing
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt

toptracks=  pd.read_csv("unique_toptracks.csv")
toptracks["TopList"]=1 #if the song is i a top list value is 1
#print(toptracks.head())
normaltracks= pd.read_csv("seperated_unique_tracks.csv")
normaltracks["TopList"]= 0
#print(normaltracks.head())
tracks=toptracks.append(normaltracks, ignore_index=True) #all tracks
#print(tracks.info())

tracks = tracks.sample(frac=1) #shuffle the data
tracks.drop(["Artist name","Song name","Track URI","Key","Mode","Year","Duration(ms)"],axis=1,inplace=True) #drop categorical features
unscaled_inputs = tracks.iloc[:,0:-1]
toplist = tracks.iloc[:,[-1]]
scaled_inputs = preprocessing.scale(unscaled_inputs)

samples_count = scaled_inputs.shape[0]
#split data into test,validation and train data
train_samples_count = int(0.8*samples_count) #80%
validation_samples_count = int(0.1*samples_count) #10%
test_samples_count = samples_count - train_samples_count - validation_samples_count #10%

# train data
train_inputs = scaled_inputs[:train_samples_count].astype(np.float)
train_targets = toplist[:train_samples_count].astype(np.int)
# validation data
validation_inputs = scaled_inputs[train_samples_count:train_samples_count+validation_samples_count].astype(np.float)
validation_targets = toplist[train_samples_count:train_samples_count+validation_samples_count].astype(np.int)
# test data
test_inputs = scaled_inputs[train_samples_count+validation_samples_count:].astype(np.float)
test_targets = toplist[train_samples_count+validation_samples_count:].astype(np.int)

#print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
#print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
#print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

# Setting the input and output sizes
input_size = 9 # count of features
output_size = 2 # count of targets
# Same hidden layer size for both hidden layers
hidden_layer_size = 50 # counts of neurons

# the model
model = tf.keras.Sequential([
    # tf.keras.layers.Dense is to output = activation(dot(input, weight) + bias)
    # hidden_layer_size and the activation function is important
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 3nd hidden layer
    # activate it with softmax
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# batch size
batch_size = 300
# maximum number of training epochs
max_epochs = 6

# fit the model
# the train, validation and test data are not iterable
history = model.fit(  train_inputs, # train inputs
                      train_targets, # train targets
                      batch_size=batch_size, # batch size
                      epochs=max_epochs, # epochs that we will train for (assuming early stopping doesn't kick in)
                      # callbacks are functions called by a task when a task is completed
                      # task here is to check if val_loss is increasing
                      #callbacks=[early_stopping], # early stopping
                      validation_data=(validation_inputs, validation_targets), # validation data
                      verbose = 2 # making sure we get enough information about the training process
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
print(plt.show())

test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))

