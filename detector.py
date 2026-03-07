def detect_event(frame_number):
    if frame_number==200:
        return {
            "event": "goal",
            "team": "Portugal",
            "player": "Ronaldo",
            "time": "00:07"
        }
    return None