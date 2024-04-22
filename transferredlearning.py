import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np

EPOCHS = 10
# Load the existing model
model = load_model('5elatermodel_202403191238.keras')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
set1 = np.load("Training_data.npy")
set2 = np.load("trainingkeys.npy")

train_data_x = set1
train_data_y = set2
TIME_STEPS = 10

# Preprocess the input data to create sequences of frames
sequences_x = []
for i in range(len(set1) - TIME_STEPS + 1):
    sequence_x = set1[i:i+TIME_STEPS]
    sequences_x.append(sequence_x)
train_data_x = np.array(sequences_x)

# Preprocess the output labels to match the number of samples
train_data_y = set2[:len(train_data_x)]
X_train, X_test, Y_train, Y_test = train_test_split(train_data_x, train_data_y,random_state=101, test_size=0.3)
# Train the model with your new dataset
try:
    history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test),verbose=1)
except KeyboardInterrupt:
    print("Stopping training dur to keyboard interupt")
    model.save('latestbalancedlatest_model.h5')

model.save('latestbalancedlatest_model.h5')