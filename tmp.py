from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def create_history(input_video,output_video,model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_video)
    track_history = defaultdict(lambda: [])
    last_positions = {}

    heatmap = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.float32)

    fps = int(cap.get(5))
    videoWriter = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, classes=2)

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x_center, y_center, width, height = box
            current_position = (float(x_center), float(y_center))

            top_left_x = int(x_center - width / 2)
            top_left_y = int(y_center - height / 2)
            bottom_right_x = int(x_center + width / 2)
            bottom_right_y = int(y_center + height / 2)

            top_left_x = max(0, top_left_x)
            top_left_y = max(0, top_left_y)
            bottom_right_x = min(heatmap.shape[1], bottom_right_x)
            bottom_right_y = min(heatmap.shape[0], bottom_right_y)

            track = track_history[track_id]
            track.append(current_position)
            if len(track) > 1200:
                track.pop(0)

            last_position = last_positions.get(track_id)
            if last_position and calculate_distance(last_position, current_position) > 5:
                heatmap[top_left_y:bottom_right_y, top_left_x:bottom_right_x] += 1

            last_positions[track_id] = current_position

            heatmap_blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)

            heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

            alpha = 0.7
            cv_dst = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)

            cv_resize = cv2.resize(cv_dst,(640,360))
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            videoWriter = cv2.VideoWriter(output_video, fourcc, fps, (cv_resize.shape[1], cv_resize.shape[0]))

        videoWriter.write(cv_resize)
        cv2.imshow("Traffic Heatmap",cv_resize)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = "yolov8s.pt"
    create_history(r"C:\Users\admin\Downloads\test.mp4",'21.mp4',model_path)
