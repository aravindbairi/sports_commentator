from commentator import generate_commentary
from detector import detect_event
from ingest import read_video
import logging

logging.basicConfig(level=logging.DEBUG)

for i, frame in enumerate(read_video("./videos/test.mov")):
    event = detect_event(i)
    if event:
        commentary = generate_commentary(event)
        print(commentary)
        break