import numpy as np
from scipy.optimize import linear_sum_assignment


class StudentTracker:
    """
    Upgrade vs original: uses the Hungarian algorithm (optimal bipartite
    matching) instead of a greedy best-score-first match.

    Why this matters for accuracy: greedy matching can lock in a locally
    good match that blocks a globally better one, causing avoidable ID
    switches in crowded frames. ID switches are bad here specifically
    because cheating "attempts" and cumulative cheating_time are tracked
    PER track_id in EventEngine - an ID switch silently resets a
    student's history. Optimal assignment reduces switch frequency.
    """

    def __init__(self, iou_threshold=0.30, max_missing=15):
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing

        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        detections = [d for d in detections if d.get("bbox") is not None]

        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        matched_tracks = set()
        matched_detections = set()

        if n_tracks and n_dets:
            cost = np.ones((n_tracks, n_dets), dtype=np.float32)

            for i, track_id in enumerate(track_ids):
                track_bbox = self.tracks[track_id]["bbox"]
                for j, detection in enumerate(detections):
                    iou = self._iou(track_bbox, detection["bbox"])
                    cost[i, j] = 1.0 - iou

            row_idx, col_idx = linear_sum_assignment(cost)

            for r, c in zip(row_idx, col_idx):
                iou = 1.0 - cost[r, c]

                if iou < self.iou_threshold:
                    continue

                track_id = track_ids[r]
                detection = detections[c]

                self.tracks[track_id].update(
                    {
                        "bbox": detection["bbox"],
                        "confidence": detection.get("confidence", 0.0),
                        "missing": 0,
                    }
                )

                if "keypoints" in detection:
                    self.tracks[track_id]["keypoints"] = detection["keypoints"]

                matched_tracks.add(track_id)
                matched_detections.add(c)

        # --------------------------------------------------
        # Create new tracks for unmatched detections
        # --------------------------------------------------
        for det_index, detection in enumerate(detections):
            if det_index in matched_detections:
                continue

            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {
                "bbox": detection["bbox"],
                "confidence": detection.get("confidence", 0.0),
                "missing": 0,
            }

            if "keypoints" in detection:
                self.tracks[track_id]["keypoints"] = detection["keypoints"]

            matched_tracks.add(track_id)
            matched_detections.add(det_index)

        # --------------------------------------------------
        # Handle temporarily missing students
        # --------------------------------------------------
        to_delete = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track["missing"] += 1
                if track["missing"] > self.max_missing:
                    to_delete.append(track_id)

        for track_id in to_delete:
            del self.tracks[track_id]

        # --------------------------------------------------
        # Return currently visible tracks
        # --------------------------------------------------
        results = []
        for track_id in matched_tracks:
            track = self.tracks[track_id]

            if track["missing"] == 0:
                result = {
                    "track_id": track_id,
                    "bbox": track["bbox"],
                    "confidence": track["confidence"],
                }

                if "keypoints" in track:
                    result["keypoints"] = track["keypoints"]

                results.append(result)

        return results

    def _iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        intersection_width = max(0, x2 - x1)
        intersection_height = max(0, y2 - y1)
        intersection = intersection_width * intersection_height

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union
