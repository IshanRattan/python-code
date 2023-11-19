
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Sequential
from preprocessing import one_hot_encode
from datasets import sign_language

import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = sign_language.load_data()
y_train_OH, y_test_OH = one_hot_encode(x_train, y_train, y_test)

# Store labels of dataset
labels = ['A', 'B', 'C']

# Print the first several training images, along with the labels
fig = plt.figure(figsize=(20, 5))
for i in range(36):
    ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))
    ax.set_title("{}".format(labels[y_train[i]]))
plt.show()


# Build Model
model = Sequential()

# First convolutional layer accepts image input
model.add(Conv2D(filters=5,
                 kernel_size=5,
                 padding='same',
                 activation='relu',
                 input_shape=(50, 50, 3)))

# Add a max pooling layer
model.add(MaxPooling2D(pool_size=(4,4),
                       strides=None,
                       padding='valid'))

# Add a convolutional layer
model.add(Conv2D(filters=15,
                 kernel_size=5,
                 padding='same',
                 activation='relu'))

# Add another max pooling layer
model.add(MaxPooling2D(pool_size=(4,4),
                       strides=None,
                       padding='valid'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# Summarize the model
model.summary()

# Compile the model
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


hist = model.fit(x_train,
                 y_train_OH,
                 validation_split=.2,
                 epochs=2,
                 batch_size=32)

# Obtain accuracy on test set
score = model.evaluate(x=x_test, 
                       y=y_test_OH,
                       verbose=0)

print('Test accuracy:', score[1])

# Get predicted probabilities for test dataset
y_probs = model.predict(x_test)

# Get predicted labels for test dataset
y_preds = np.argmax(y_probs, axis=1)

# Indices corresponding to test images which were mislabeled
bad_test_idxs = np.where(y_preds!=y_test)[0]

print('===== Mislabeled Images =====')
print(bad_test_idxs)

# # Print mislabeled examples
# fig = plt.figure(figsize=(25,4))
# for i, idx in enumerate(bad_test_idxs):
#     ax = fig.add_subplot(2, np.ceil(len(bad_test_idxs)/2), i + 1, xticks=[], yticks=[])
#     ax.imshow(np.squeeze(x_test[idx]))
#     ax.set_title("{} (pred: {})".format(labels[y_test[idx]], labels[y_preds[idx]]))
