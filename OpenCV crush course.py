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
        if start_time_in_msec <= timestamp <= end_time_in_msec:
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
    # region read frame
    has_frame, frame = vs.read()
    if not has_frame:
        break
    # endregion

    # region change brightness and contrast
    # adjusting brightness:
    matrix = np.ones(frame.shape, dtype="uint8") * 50
    frame = cv2.add(frame, matrix)
    # adjusting contrast:
    matrix = np.ones(frame.shape) * 1.3
    frame = np.uint8(np.clip(cv2.multiply(np.float64(frame), matrix), 0, 255))

    # endregion

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_frame_thresh_adp = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 4)
    # gray_frame_thresh_adp = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9,2)

    # region canny filtering
    gray_frame = cv2.blur(gray_frame, (2, 2))
    gray_frame = cv2.Canny(gray_frame, 75, 80)

    cv2.imshow("grayframe", gray_frame)
    # cv2.imshow("grayframe_thresh_adp", gray_frame_thresh_adp)

    pressed_key = cv2.waitKey(1)  # waits 1 mSec
    if pressed_key == 27 or pressed_key == ord("q") or pressed_key == ord("Q"):  # Esc pressed
        alive = False

# closing the video properly:
vs.release()
cv2.destroyAllWindows()
exit()
