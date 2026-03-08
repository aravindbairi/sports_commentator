from collections import deque

import cv2


def read_sliding_window(video_path,T=16,stride=8,max_windows=50):
    cap = cv2.VideoCapture(video_path)
    frame_buffer = deque(maxlen=T)
    frame_count =0
    windows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count+=1
        frame_buffer.append(frame)
        if len(frame_buffer) == T and frame_count % stride == 0:
            frames = list(frame_buffer)
            windows.append((frame_count,frames))
            if len(windows)>max_windows:
                break
    cap.release()
    return windows