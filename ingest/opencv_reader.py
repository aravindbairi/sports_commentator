from collections import deque

import cv2

from detector import infer_clip


def sliding_infer(video_path,T=16,stride=8,debug=False):
    cap = cv2.VideoCapture(video_path)
    frame_buffer = deque(maxlen=T)
    frame_count =0;
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count+=1
        frame_buffer.append(frame)
        if len(frame_buffer) == T and frame_count % stride == 0:
            frames = list(frame_buffer)
            preds = infer_clip(frames,topk=5)
            yield {"frames":frames,"preds":preds}
    cap.release()
def read_video(path):
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()