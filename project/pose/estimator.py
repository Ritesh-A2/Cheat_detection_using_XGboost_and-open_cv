class PoseEstimator:
    """
    Extracts normalized keypoints from a single YOLO-Pose result.
    No separate model - yolo11n-pose.pt already produces keypoints
    alongside detection boxes, so this is a pure post-processing step.
    """

    def extract(self, result):
        if result.keypoints is None:
            return []
        return result.keypoints.xyn.tolist()
