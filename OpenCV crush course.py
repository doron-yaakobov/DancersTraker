'''
Track the two dancers.
Time frame: 3:18 - 4:33
'''

import cv2
import numpy as np
import moviepy.editor as mp_editor


# def crop_video(start_time_in_msec: int = 198000, end_time_in_msec: int = 273000,
#                src_file: str = 'data/video/LaLaLand_A_lovely_night_scene.mp4',
#                dst_file: str = 'data/video/cropped_video'):
#     # Open the video
#     vs = cv2.VideoCapture(src_file)
#     # Set the video writer
#     frame_width = int(vs.get(3))
#     frame_height = int(vs.get(4))
#
#     fourcc = cv2.VideoWriter.fourcc(*'XVID')
#     out = cv2.VideoWriter(f'{dst_file}.mp4', fourcc, 10, (frame_width, frame_height))
#
#     # Read and write the frames
#     while True:
#         ret, frame = vs.read()
#         if not ret:
#             break
#
#         # Get the current timestamp
#         timestamp = vs.get(cv2.CAP_PROP_POS_MSEC)
#
#         # Check if the current timestamp is within the desired range
#         if start_time_in_msec <= timestamp <= end_time_in_msec:
#             out.write(frame)
#
#         # Check if the current timestamp is past the desired range
#         if timestamp > end_time_in_msec:
#             break
#
#     # Release the video writer and close the video
#     out.release()
#     vs.release()
#

def crop_video(start_time_in_sec: int = 2 * 60 + 55, end_time_in_sec: int = 4 * 60 + 33,
               src_file: str = 'data/video/LaLaLand_A_lovely_night_scene.mp4',
               dst_file: str = 'data/video/cropped_video_ver_3.mp4'):
    src = mp_editor.VideoFileClip(src_file)
    dst = src.subclip(start_time_in_sec, end_time_in_sec)
    dst.write_videofile(dst_file)
    src.reader.close()
    src.audio.reader.close_proc()
    return 0


def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)


def canny_filter(bgr_frame, dtype: str = "uint8", brightness_factor: int = 50, contrast_factor: int = 1.3,
                 blurring_area=(2, 2), canny_min: int = 130, canny_max: int = 140):
    def adjust_brightness(bgr_frame, dtype, factor: int = brightness_factor):
        matrix = np.ones(bgr_frame.shape, dtype=dtype) * factor
        return cv2.add(bgr_frame, matrix)

    def adjust_contrast(bgr_frame, factor: int = contrast_factor):
        matrix = np.ones(bgr_frame.shape) * factor
        return np.uint8(np.clip(cv2.multiply(np.float64(bgr_frame), matrix), 0, 255))

    bgr_frame = adjust_brightness(bgr_frame, dtype, brightness_factor)
    bgr_frame = adjust_contrast(bgr_frame, contrast_factor)
    gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    gray_frame_thresh_adp = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 4)
    blur_frame = cv2.blur(gray_frame_thresh_adp, blurring_area)
    return cv2.Canny(blur_frame, canny_min, canny_max)


if __name__ == "__main__":
    # crop_video()

    output_video = "data/video/cv2.trackerCSRT_output_ver3"
    vs = cv2.VideoCapture('data/video/cropped_video_ver_3.mp4')
    tracker = cv2.legacy.TrackerCSRT.create()
    frame_width = int(vs.get(3))
    frame_height = int(vs.get(4))
    # out = cv2.VideoWriter(f'{output_video}.mp4', cv2.VideoWriter.fourcc(*'XVID'), 10, (frame_width, frame_height))

    # region init tracker
    has_frame, frame = vs.read()
    bbox = [725, 262, 215, 650]
    drawRectangle(frame, bbox)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cv2.destroyWindow("frame")
    # endregion
    cv2.imwrite("data/image/bboxed_first_dancer_frame.png", frame)
    adjusted_frame = canny_filter(frame)
    ok = tracker.init(adjusted_frame, bbox)

    print(f"traker init stauts:\t{ok}")

    alive = True
    while alive:
        has_frame, frame = vs.read()
        if not has_frame:
            print("No frame! breaking")
            break
        adjusted_frame = canny_filter(frame)

        timer = cv2.getTickCount()
        ok, bbox = tracker.update(adjusted_frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok:
            drawRectangle(frame, bbox)
        else:
            print("Failed to track.")

        cv2.imshow("frame", frame)
        # out.write(frame)

        pressed_key = cv2.waitKey(1)  # waits 1 mSec
        if pressed_key == 27 or pressed_key == ord("q") or pressed_key == ord("Q"):  # Esc pressed
            alive = False

    # closing the video properly:
    # out.release()
    vs.release()
    cv2.destroyAllWindows()

    exit()
