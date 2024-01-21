import cv2
import numpy as np
import matplotlib.pyplot as plt

vs = cv2.VideoCapture('data/LaLaLand_A_lovely_night_scene.mp4')
alive = True
while alive:
    _, frame = vs.read()
    print(f"Image size is {frame.shape}")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"Gray Image size is {gray_frame.shape}")
    print(f"Data type is {gray_frame.dtype}")

    cv2.imshow("grayframe", gray_frame)
    pressed_key = cv2.waitKey(1)  # waits 1 mSec
    if pressed_key == 27 or pressed_key == ord("q"):  # Esc pressed
        alive = False

# closing the video properly:
vs.release()
cv2.destroyAllWindows()
exit()
