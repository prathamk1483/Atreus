import numpy as np
import cv2
from mss import mss
import time
from getkeys import key_check

bbox = {'top': 0, 'left': 0, 'width': 700, 'height': 480}

sct = mss()
ft = time.time()
count = 0

w = [1, 0, 0, 0, 0]
a = [0, 1, 0, 0, 0]
s = [0, 0, 1, 0, 0]
d = [0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 1]

def keys_to_output(keys):
    if "W" in keys:
        output = w
    elif "A" in keys:
        output = a
    elif "S" in keys:
        output = s
    elif "D" in keys:
        output = d
    else:
        output = nk
    return np.array(output)

training_data = []
training_keys = []

while True:
    screen = np.array(sct.grab(bbox))  # Capture screen
    keys = key_check()
    output = keys_to_output(keys)
    
    # Convert RGBA to RGB if needed
    if screen.shape[2] == 4:  # Check if image is RGBA
        screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
    
    screen = cv2.resize(screen, (160, 120))
    training_data.append(screen)
    training_keys.append(output)
    
    print(len(training_data))
    
    if "R" in keys:
        print("Paused")
        time.sleep(10)
    elif len(training_data) % 500000 == 0 or 'T' in keys:
        print(len(training_data), "     ", keys)
        np.save("Training_data.npy", training_data)
        break

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break

np.save("trainingkeys.npy", training_keys, allow_pickle=True)
