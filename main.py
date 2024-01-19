'''
Track the two dancers.
Time frame: 3:18 - 4:33
'''

import cv2
import numpy as np

cap = cv2.VideoCapture('data/LaLaLand_A_lovely_night_scene.mp4')

while True:
    # getting frames one by one out of the video
    _, frame = cap.read()

    cv2.imshow("Frame", frame)

    # cv2.waitKey(0)  # waits for a key to be pushed
    key = cv2.waitKey(1)  # waits 1 mSec between frames
    if key == 27:   # if "Esc" is typed:
        break

# closing the video properly:
cap.release()
cv2.destroyAllWindows()