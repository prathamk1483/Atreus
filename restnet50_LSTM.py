from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
import keras
from keras.layers import TimeDistributed


# Load pre-trained ResNet50 without top layers


# Define restnet50_LSTM function
def restnet50_LSTM(WIDTH, HEIGHT, LR):
    # Define LSTM model
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(WIDTH, HEIGHT, 3))
    for layer in resnet.layers:
        layer.trainable = False

    model = Sequential()
    model.add(TimeDistributed(resnet, input_shape=(10, WIDTH, HEIGHT, 3)))  # TimeDistributed to apply ResNet to each frame
    model.add(TimeDistributed(Flatten()))  # Flatten output of ResNet
    model.add(LSTM(128))  # LSTM layer to capture temporal dependencies
    model.add(Dense(64, activation='relu'))  # Additional dense layer
    model.add(Dense(5, activation='softmax'))  # Output layer

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=LR),
                  metrics=['accuracy'])
    
    return model
