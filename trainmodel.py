import os
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
from restnet50_LSTM import restnet50_LSTM

# Load data
set1 = np.load("Training_data.npy")
set2 = np.load("trainingkeys.npy")

# Separate input and output arrays
train_data_x = set1
train_data_y = set2

# Define the number of frames per sequence
TIME_STEPS = 10

# Preprocess the input data to create sequences of frames
sequences_x = []
for i in range(len(set1) - TIME_STEPS + 1):
    sequence_x = set1[i:i+TIME_STEPS]
    sequences_x.append(sequence_x)
train_data_x = np.array(sequences_x)

# Preprocess the output labels to match the number of samples
train_data_y = set2[:len(train_data_x)]

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(train_data_x, train_data_y, random_state=101, test_size=0.3)

# Define model parameters
WIDTH = 160
HEIGHT = 120
LR = 1e-3

# Create the model
model = restnet50_LSTM(train_data_x.shape[2], train_data_x.shape[3], LR)

# Train the model
EPOCHS = 5
try:
    history = model.fit(X_train, Y_train, epochs=EPOCHS, validation_data=(X_test, Y_test), verbose=1)
except KeyboardInterrupt:
    print("Stopping training due to keyboard interrupt")
    model.save('balancedinterrupted.keras')
else:
    # Save the trained model with a timestamp to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%H%M%S")
    model_filename = f'5elatermodel_{timestamp}.keras'
    model.save(model_filename)
