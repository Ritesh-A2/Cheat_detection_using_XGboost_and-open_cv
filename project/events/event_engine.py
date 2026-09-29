from collections import deque


class EventEngine:
    """
    Upgrades vs original:
      1. Score smoothing: classifier scores are noisy frame-to-frame
         (a single bad pose read can dip below threshold). A rolling
         average over `smoothing_window` frames is thresholded instead
         of the raw per-frame score, cutting spurious single-frame
         false positives.
      2. cheating_time decay: previously cheating_time only ever
         increased, so ANY brief past incident permanently kept a
         student flagged/counted for the rest of the session. Now,
         while NOT currently cheating, cheating_time decays at
         `decay_per_sec`. This keeps "attempts" reflecting sustained
         behaviour rather than a single old glance.
    """

    def __init__(self, threshold=0.70, smoothing_window=5, decay_per_sec=0.5):
        self.threshold = threshold
        self.smoothing_window = smoothing_window
        self.decay_per_sec = decay_per_sec

        self.students = {}

    def _new_state(self, timestamp, score):
        return {
            "cheating": False,
            "last_timestamp": timestamp,
            "cheating_time": 0.0,
            "attempts": 0,
            "last_score": score,
            "score_history": deque(maxlen=self.smoothing_window),
        }

    def update(self, track_id, score, timestamp):
        if score is None:
            return None

        if track_id not in self.students:
            self.students[track_id] = self._new_state(timestamp, score)

        state = self.students[track_id]

        state["score_history"].append(score)
        smoothed_score = sum(state["score_history"]) / len(state["score_history"])

        # LOW score -> cheating, HIGH score -> normal
        cheating = smoothed_score < self.threshold

        previous_time = state["last_timestamp"]
        elapsed = timestamp - previous_time

        if elapsed < 0:
            elapsed = 0.0
        if elapsed > 1.0:
            elapsed = 1.0

        if cheating:
            state["cheating_time"] += elapsed
        else:
            state["cheating_time"] = max(
                0.0, state["cheating_time"] - self.decay_per_sec * elapsed
            )

        new_attempts = int(state["cheating_time"])

        event = None
        if new_attempts > state["attempts"]:
            state["attempts"] = new_attempts

            event = {
                "type": "CHEATING_ATTEMPT",
                "track_id": track_id,
                "score": smoothed_score,
                "attempts": state["attempts"],
                "cheating_time": state["cheating_time"],
            }

        state["cheating"] = cheating
        state["last_timestamp"] = timestamp
        state["last_score"] = smoothed_score

        return event

    def get_student_state(self, track_id):
        state = self.students.get(track_id)
        if state is None:
            return None

        return {
            "track_id": track_id,
            "cheating": state["cheating"],
            "attempts": state["attempts"],
            "cheating_time": state["cheating_time"],
            "last_score": state["last_score"],
        }

    def remove_track(self, track_id):
        self.students.pop(track_id, None)

    def reset(self):
        self.students.clear()
