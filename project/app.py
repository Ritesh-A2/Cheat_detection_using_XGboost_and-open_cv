from config.settings import (
    MODEL_YOLO,
    MODEL_XGB,
    YOLO_CONF,
    YOLO_IOU,
    MAX_VIDEOS,
    EVENT_THRESHOLD,
    SCORE_SMOOTHING_WINDOW,
    CHEATING_TIME_DECAY_PER_SEC,
    TRACK_IOU_THRESHOLD,
    TRACK_MAX_MISSING,
    DEVICE,
    USE_HALF_PRECISION,
)

from detection.detector import PersonDetector
from pose.estimator import PoseEstimator
from classifier.xgb_model import XGBClassifier
from tracking.tracker import StudentTracker
from events.event_engine import EventEngine

from inference.worker import InferenceWorker
from ui.tkinter_app import gui
from utils.logging import get_logger

logger = get_logger("app")


def create_ai_engine():
    """
    Create and connect all AI modules.

    Models are loaded only once here.
    GUI does not load or manage AI models.
    """

    logger.info(f"Initializing AI engine on device={DEVICE}, half={USE_HALF_PRECISION}")

    # --------------------------------------------------
    # AI MODELS
    # --------------------------------------------------

    detector = PersonDetector(
        MODEL_YOLO,
        confidence=YOLO_CONF,
        iou=YOLO_IOU,
        device=DEVICE,
        half=USE_HALF_PRECISION,
    )

    classifier = XGBClassifier(
        MODEL_XGB,
        device=DEVICE,
    )

    # --------------------------------------------------
    # POSE EXTRACTION
    # --------------------------------------------------

    pose_estimator = PoseEstimator()

    # --------------------------------------------------
    # INDEPENDENT TRACKER FOR EACH VIDEO
    # --------------------------------------------------

    trackers = [
        StudentTracker(
            iou_threshold=TRACK_IOU_THRESHOLD,
            max_missing=TRACK_MAX_MISSING,
        )
        for _ in range(MAX_VIDEOS)
    ]

    # --------------------------------------------------
    # INDEPENDENT EVENT ENGINE FOR EACH VIDEO
    # --------------------------------------------------

    event_engines = [
        EventEngine(
            threshold=EVENT_THRESHOLD,
            smoothing_window=SCORE_SMOOTHING_WINDOW,
            decay_per_sec=CHEATING_TIME_DECAY_PER_SEC,
        )
        for _ in range(MAX_VIDEOS)
    ]

    # --------------------------------------------------
    # CENTRAL INFERENCE WORKER
    # --------------------------------------------------

    worker = InferenceWorker(
        detector=detector,
        pose_estimator=pose_estimator,
        classifier=classifier,
        trackers=trackers,
        event_engines=event_engines,
    )

    logger.info("AI engine ready.")

    return worker


def main():
    worker = create_ai_engine()
    gui(worker)


if __name__ == "__main__":
    main()
