import cv2
def read_video(path):
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()