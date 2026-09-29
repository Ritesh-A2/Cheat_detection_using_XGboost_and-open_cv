class InferenceWorker:
    """
    Upgrade vs original: classification is now batched across ALL
    tracked people from ALL active video slots into a single
    classifier.classify_batch() call per inference cycle, instead of
    one classify() call per person. Combined with the already-batched
    YOLO detection call, this means each cycle does at most:
      - 1 YOLO forward pass (all video frames)
      - 1 XGBoost predict call (all detected people, grouped by
        keypoint count)
    regardless of how many students are on screen across the 4 videos.
    """

    def __init__(self, detector, pose_estimator, classifier, trackers, event_engines):
        self.detector = detector
        self.pose = pose_estimator
        self.classifier = classifier

        self.trackers = trackers
        self.events = event_engines

    def process_batch(self, batch):
        if not batch:
            return []

        # --------------------------------------------------
        # 1. ONE BATCHED YOLO-Pose INFERENCE ACROSS ALL VIDEOS
        # --------------------------------------------------
        frames = [item["frame"] for item in batch]
        results = self.detector.detect(frames)

        per_item_tracks = []

        # --------------------------------------------------
        # 2. DETECTION + TRACKING PER VIDEO
        # --------------------------------------------------
        for item, result in zip(batch, results):
            slot = item["slot"]
            tracker = self.trackers[slot]

            keypoints = self.pose.extract(result)

            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for i, bbox in enumerate(boxes):
                    detection = {
                        "bbox": bbox.tolist(),
                        "confidence": float(confidences[i]),
                    }

                    # YOLO detection index and keypoint index correspond
                    # to the same person for ultralytics pose models.
                    if i < len(keypoints):
                        detection["keypoints"] = keypoints[i]

                    detections.append(detection)

            tracks = tracker.update(detections)
            per_item_tracks.append((item, tracks))

        # --------------------------------------------------
        # 3. ONE BATCHED XGBOOST CALL ACROSS ALL PEOPLE / ALL VIDEOS
        # --------------------------------------------------
        all_keypoints = []
        index_map = []  # (item_idx, track_idx)

        for item_idx, (item, tracks) in enumerate(per_item_tracks):
            for track_idx, track in enumerate(tracks):
                kp = track.get("keypoints")
                if kp is not None:
                    all_keypoints.append(kp)
                    index_map.append((item_idx, track_idx))

        if all_keypoints:
            classifications = self.classifier.classify_batch(
                all_keypoints, threshold=0.70
            )
            for (item_idx, track_idx), classification in zip(
                index_map, classifications
            ):
                track = per_item_tracks[item_idx][1][track_idx]
                track["score"] = classification["score"]
                track["is_suspicious"] = classification["is_suspicious"]

        # --------------------------------------------------
        # 4. EVENT GENERATION + RESULT ASSEMBLY
        # --------------------------------------------------
        output = []

        for item, tracks in per_item_tracks:
            slot = item["slot"]
            frame_no = item["frame_no"]
            timestamp = item["timestamp"]
            event_engine = self.events[slot]

            frame_events = []

            for track in tracks:
                if "score" not in track:
                    continue

                event = event_engine.update(
                    track["track_id"], track["score"], timestamp
                )

                if event is not None:
                    event["slot"] = slot
                    event["frame_no"] = frame_no
                    frame_events.append(event)

            output.append(
                {
                    "slot": slot,
                    "frame_no": frame_no,
                    "frame": item["frame"],
                    "tracks": tracks,
                    "events": frame_events,
                }
            )

        return output
