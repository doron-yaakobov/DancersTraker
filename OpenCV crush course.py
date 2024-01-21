'''
Track the two dancers.
Time frame: 3:18 - 4:33
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_video(start_time_in_msec: int = 198000, end_time_in_msec: int = 273000,
               src_file: str = 'data/LaLaLand_A_lovely_night_scene.mp4', dst_file: str = 'data/cropped_video'):
    # Open the video
    vs = cv2.VideoCapture(src_file)
    # Set the video writer
    frame_width = int(vs.get(3))
    frame_height = int(vs.get(4))

    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(f'{dst_file}.mp4', fourcc, 10, (frame_width, frame_height))

    # Read and write the frames
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        # Get the current timestamp
        timestamp = vs.get(cv2.CAP_PROP_POS_MSEC)

        # Check if the current timestamp is within the desired range
        if timestamp >= start_time_in_msec and timestamp <= end_time_in_msec:
            out.write(frame)

        # Check if the current timestamp is past the desired range
        if timestamp > end_time_in_msec:
            break

    # Release the video writer and close the video
    out.release()
    vs.release()


# crop_video()

vs = cv2.VideoCapture('data/cropped_video.mp4')
alive = True
while alive:
    _, frame = vs.read()
    print(f"Image size is {frame.shape}")

    # cv2.line(frame, (200, 100), (400, 100), (0, 255, 255), thickness=5, lineType=cv2.LINE_AA)
    # cv2.rectangle(frame, (500, 100), (700, 600), (0, 255, 255), thickness=5, lineType=cv2.LINE_AA)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"Gray Image size is {gray_frame.shape}")
    print(f"Data type is {gray_frame.dtype}")

    # gray_frame_thresh_adp = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 4)
    gray_frame_thresh_adp = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    cv2.imshow("grayframe", gray_frame)
    cv2.imshow("grayframe", gray_frame_thresh_adp)
    pressed_key = cv2.waitKey(1)  # waits 1 mSec
    if pressed_key == 27 or pressed_key == ord("q") or pressed_key == ord("Q"):  # Esc pressed
        alive = False

# closing the video properly:
vs.release()
cv2.destroyAllWindows()
exit()
