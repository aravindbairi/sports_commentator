import cv2

from detector.infer import Detector
from ingest.opencv_reader import read_sliding_window

# logging.basicConfig(level=logging.DEBUG)

def main():
    detector = Detector(checkpoint="./checkpoints/best_model.pth")
    windows = read_sliding_window(".idea/test.mov", T=16, stride=8,max_windows=50)
    for frame_idx, clip in windows[:20]:
        out = detector.infer_clip(clip)
        print(f"frame={frame_idx} -- time={out['time']:.1f}s topk={out['topk']}")
    print("Done")

if __name__ == "__main__":
    main()