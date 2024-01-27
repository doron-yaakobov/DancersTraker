# region imports
import numpy as np
import datetime
import cv2
import torch
from absl import app, flags, logging
from absl.flags import FLAGS
from deep_sort_realtime.deepsort_tracker import DeepSort
from super_gradients.training import models
import mediapipe as mp
import moviepy.editor as mp_editor
from super_gradients.common.object_names import Models

# endregion


# region Define command line flags
flags.DEFINE_float('conf', 0.50, 'confidence threshhold')
flags.DEFINE_integer('class_id', 0, 'class id 0 for person check coco.names for others')
flags.DEFINE_integer('blur_id', None, 'class id to blurring the object')


# endregion

# region funcs
def crop_video(start_time_in_sec: int = 2 * 60 + 55, end_time_in_sec: int = 4 * 60 + 33,
               src_file: str = 'data/video/LaLaLand_A_lovely_night_scene.mp4',
               dst_file: str = 'data/video/cropped_video_ver_4.mp4'):
    src = mp_editor.VideoFileClip(src_file)
    dst = src.subclip(start_time_in_sec, end_time_in_sec)
    dst.write_videofile(dst_file)
    src.reader.close()
    src.audio.reader.close_proc()
    return 0


def run_YOLO_NAS_Tracker(video_cap, writer, model: str, classes_path: str, male_dancer_track_id: str,
                         female_dancer_track_id: str):
    # region Init YOLO-NAS
    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=50)
    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Load the YOLO model
    model = models.get(model, pretrained_weights="coco").to(device)
    # Load the COCO class labels the YOLO model was trained on
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")
    # Create a list of random colors to represent each class
    np.random.seed(42)  # to get the same colors
    colors = np.random.randint(0, 255, size=(len(class_names), 3))  # (80, 3)
    # endregion

    while True:
        # Start time to compute the FPS
        start = datetime.datetime.now()

        is_frame, frame = video_cap.read()
        if not is_frame:
            print("End of the video file...")
            break

        # region YOLO-NAS on frame
        detect = next(iter(model.predict(frame, iou=0.5, conf=FLAGS.conf)))
        # Extract the bounding box coordinates, confidence scores, and class labels from the detection results
        bboxes_xyxy = torch.from_numpy(detect.prediction.bboxes_xyxy).tolist()
        confidence = torch.from_numpy(detect.prediction.confidence).tolist()
        labels = torch.from_numpy(detect.prediction.labels).tolist()
        # Combine the bounding box coordinates and confidence scores into a single list
        concate = [sublist + [element] for sublist, element in zip(bboxes_xyxy, confidence)]
        # Combine the concatenated list with the class labels into a final prediction list
        final_prediction = [sublist + [element] for sublist, element in zip(concate, labels)]
        results = []
        # Loop over the detections
        for data in final_prediction:
            # region filter out weak \ un-relevant detections
            confidence = data[4]
            if FLAGS.class_id == None:
                if float(confidence) < FLAGS.conf:
                    continue
            else:
                if ((int(data[5] != FLAGS.class_id)) or (float(confidence) < FLAGS.conf)):
                    continue
            # endregion

            # Define BBox
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            # Add the bounding box (x, y, w, h), confidence, and class ID to the results list
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
        # endregion

        tracks = tracker.update_tracks(results, frame=frame)
        # Loop over the tracks
        for track in tracks:
            # If the track is not confirmed, ignore it
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            color = colors[class_id]
            B, G, R = int(color[0]), int(color[1]), int(color[2])

            if track_id == male_dancer_track_id:
                text = f"Male Dancer"
            elif track_id == female_dancer_track_id:
                text = "Female Dancer"
            else:
                text = f"Unknown"

            # Apply Gaussian Blur
            if FLAGS.blur_id is not None and class_id == FLAGS.blur_id:
                if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 5)

            # Draw bounding box and text on the frame
            if text != "Unknown":
                B, G, R = (193, 182, 255) if text == "Female Dancer" else (B, G, R)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
                cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # End time to compute the FPS
        end = datetime.datetime.now()
        # Show the time it took to process 1 frame
        print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
        # Calculate the frames per second and draw it on the frame
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
        cv2.imshow("Frame", frame)
        writer.write(frame)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break


def init_writer(video_cap, output: str, output_format: str = 'MP4V'):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter.fourcc(*output_format)
    return cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))


# endregion

def main(_argv):
    # region config and setup
    ROOT_DIR = r"C:\Users\dorony\PycharmProjects\DancersTraker"
    model = "yolo_nas_l"
    classes_path = "./configs/coco.names"
    scene_start_time = 3 * 60 + 18
    scene_end_time = 4 * 60 + 33
    src_video = ROOT_DIR + r"\data\Video\LaLaLand_A_lovely_night_scene.mp4"
    dst = ROOT_DIR + r"\\data\Video\output\requested_scene_cropped.mp4"
    dst_part_1 = ROOT_DIR + r"\data\Video\output\requested_scene_cropped_pt_1.mp4"
    dst_part_2 = ROOT_DIR + r"\data\Video\output\requested_scene_cropped_pt_2.mp4"
    dst_part_3 = ROOT_DIR + r"\data\Video\output\requested_scene_cropped_pt_3.mp4"
    output = ROOT_DIR + r"\data\Video\output\Tracked_Dancers.mp4"
    crop_video(scene_start_time, scene_end_time, src_video, dst)
    crop_video(0, 36, dst, dst_part_1)
    crop_video(36, 58, dst, dst_part_2)
    crop_video(58, 1 * 60 + 15, dst, dst_part_3)
    writer_initialized = False
    # endregion

    for curr_video in [dst_part_1, dst_part_2, dst_part_3]:
        video_cap = cv2.VideoCapture(curr_video)

        if not writer_initialized:
            writer = init_writer(video_cap, output)
            writer_initialized = True

        male_track_id = "1" if curr_video in [dst_part_3, dst_part_2] else "2"
        female_track_id = "2" if curr_video in [dst_part_3, dst_part_2] else "1"
        run_YOLO_NAS_Tracker(video_cap, writer, model, classes_path, male_track_id, female_track_id)

        video_cap.release()

    writer.release()
    cv2.destroyAllWindows()
    exit()


if __name__ == '__main__':
    app.run(main)
