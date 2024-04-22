import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input  # Needed for defining the input layer shape

def alexnet(width, height, LR):
    """
    Creates and returns an AlexNet model with the specified input shape and learning rate.

    Args:
        width (int): Width of the input images.
        height (int): Height of the input images.
        LR (float): Learning rate to be used during training.

    Returns:
        tensorflow.keras.Model: The compiled AlexNet model.
    """

    # Create an Input layer with the specified shape
    input_layer = Input(shape=(width, height, 4))  # Adjust for grayscale images

    # Instantiate an empty model
    model = Sequential()

    # Add the input layer to the model
    model.add(input_layer)

    # 1st Convolutional Layer (Adjusted padding to avoid negative output size issue)
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 3rd, 4th, and 5th Convolutional Layers
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Flatten the output before the fully connected layers
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1000))  # Adjust to the number of classes in your dataset
    model.add(Activation('relu'))  # Consider replacing with 'softmax' if necessary

    # Output Layer (Adjusted to match your dataset)
    model.add(Dense(3))  # Adjust the number of units to match the number of classes
    model.add(Activation('softmax'))  # Use 'softmax' for multi-class classification

    # Compile the model (Ensure compatibility with your dataset and requirements)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=LR),
                  metrics=['accuracy'])

    return model
