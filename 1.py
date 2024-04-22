import numpy as np
import cv2
from mss import mss
import time
from keras.models import load_model
from directkeys import PressKey, ReleaseKey, W, A, S, D
from getkeys import key_check

bbox = {'top': 0, 'left': 0, 'width': 700, 'height': 480}
sct = mss()
ft = time.time()
count = 0
t_time = 0.06


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    time.sleep(0.09)
    ReleaseKey(W)


def left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    time.sleep(t_time)
    ReleaseKey(W)
    ReleaseKey(A)


def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    time.sleep(t_time)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse():
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(S)


# def reverseleft():
#     PressKey(S)
#     PressKey(A)
#     ReleaseKey(W)
#     ReleaseKey(D)
#     time.sleep(t_time)
#     ReleaseKey(S)
#     ReleaseKey(A)


# def reverseright():
#     PressKey(S)
#     PressKey(D)
#     ReleaseKey(W)
#     ReleaseKey(A)
#     time.sleep(t_time)
#     ReleaseKey(S)
#     ReleaseKey(D)


# Function to define a region of interest (not used in current implementation)
def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


# Function to process the image (not used in current implementation)
def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, 100, 200)
    return processed_img

paused = False
model = load_model('5elatermodel_202403191238.keras')
while True:
    frames = []
    for _ in range(10):
        screen = np.array(sct.grab(bbox))
        screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)
        screen = cv2.resize(screen, (160, 120))
        frames.append(screen)

    # Stack the frames along a new axis to create a batch of 10 frames
    batch = np.stack(frames, axis=0)

    # Add a batch dimension of size 1 to match the model's input shape
    resized_batch = np.expand_dims(batch, axis=0)

    # Preprocess the batch (you might need additional preprocessing here)
    resized_batch = resized_batch.astype(np.uint8)

    # Make prediction
    prediction = model.predict(resized_batch)

    idx = int(np.argmax(prediction[0]))
    turn_thresh = 0.2

    if idx == 1:
        left()
        print("left")
    elif idx == 0:
        straight()
        print("Straight")
    elif idx == 2:
        reverse()
        print("reverse")
    elif idx == 3:
        right()
        print("right")
    else:
        print("Something went wrong")

    keys = key_check()

    if 'T' in keys:
        if paused:
            paused = False
            time.sleep(1)
        else:
            paused = True
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            time.sleep(10)

    if time.time() - ft >= 1000:
        print(f"We have {count} FPS")
        count = 0
