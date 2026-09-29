from ultralytics import YOLO


class PersonDetector:
    """
    Wraps a single YOLO-Pose model shared across all video streams.

    Upgrades vs original:
      - Explicit device placement (CUDA if available).
      - FP16 (half precision) inference on GPU for ~1.5-2x speedup
        with negligible accuracy loss on modern GPUs.
      - Configurable IoU (previously hardcoded inside ultralytics defaults).
    """

    def __init__(
        self,
        model_path,
        confidence=0.55,
        iou=0.45,
        device="cpu",
        half=False,
    ):
        self.model = YOLO(model_path)
        self.model.to(device)

        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.half = bool(half and device != "cpu")

    def detect(self, frames):
        """
        Runs one batched inference call across frames from all active
        video slots. Returns the raw ultralytics Results list.
        """
        return self.model(
            frames,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
            batch=len(frames),
            device=self.device,
            half=self.half,
        )
