import cv2
import os

def extract_frames(video_path, config):
    frame_interval = config["frame_interval"]
    output_dir = "frames"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps * frame_interval)
    count = 0
    saved = 0
    paths = []

    while True:
        ret, frame = cap.read()
        if not ret or saved >= config["max_frame_count"]:
            break
        if count % frame_skip == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            paths.append(frame_path)
            saved += 1
        count += 1
    cap.release()
    return paths