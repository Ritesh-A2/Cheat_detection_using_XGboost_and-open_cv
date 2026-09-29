import time

import cv2

# FIX: original imported "utils.queue" (singular) which does not exist -
# the actual module is "utils.queues" (plural). This raised
# ModuleNotFoundError the moment VideoReader was imported.
from utils.queues import safe_queue_put


class VideoReader:
    def __init__(self, slot, source, frame_queue, stop_event, display_size=(640, 360)):
        self.slot = slot
        self.source = source
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.display_size = display_size

    def run(self):
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        interval = 1.0 / fps
        frame_no = 0

        while not self.stop_event.is_set():
            start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, self.display_size)

            # Video-relative timestamp (not wall clock), used by
            # EventEngine's elapsed-time accumulation.
            timestamp = frame_no / fps

            item = {
                "slot": self.slot,
                "frame_no": frame_no,
                "frame": frame,
                "timestamp": timestamp,
                "fps": fps,
            }

            safe_queue_put(self.frame_queue, item, self.stop_event)

            frame_no += 1

            elapsed = time.perf_counter() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

        cap.release()

        try:
            self.frame_queue.put_nowait(("END", self.slot))
        except Exception:
            pass
