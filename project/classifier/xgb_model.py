import numpy as np
import xgboost as xgb


class XGBClassifier:
    """
    Upgrades vs original:
      - GPU predictor ("device": "cuda") when available, falls back
        silently to CPU predictor if the installed xgboost/model
        doesn't support it.
      - Batched prediction: classify_batch() scores ALL people across
        ALL active video slots in a single DMatrix/predict() call per
        inference cycle, instead of one Python-level XGBoost call per
        person. This removes per-call overhead that scales badly once
        a classroom has many students across 4 videos.
      - predict()/classify() kept for backward compatibility, now
        implemented on top of the batched path.
    """

    def __init__(self, model_path, device="cpu"):
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        try:
            self.model.set_param({"device": device})
        except Exception:
            # Older xgboost versions / CPU-only builds: ignore and
            # let xgboost use its default predictor.
            pass

        self.feature_cache = {}

    def _feature_names(self, n):
        names = self.feature_cache.get(n)
        if names is None:
            names = []
            for i in range(n):
                names.append(f"x{i}")
                names.append(f"y{i}")
            self.feature_cache[n] = names
        return names

    def predict_batch(self, keypoints_list):
        """
        keypoints_list: list of per-person keypoint lists (each a list
        of [x, y] pairs). Returns a list of float scores (or None for
        empty keypoint entries), same length/order as input.
        """
        results = [None] * len(keypoints_list)

        # Group by keypoint count since DMatrix requires a fixed
        # feature-vector length per batch.
        groups = {}
        for idx, kp in enumerate(keypoints_list):
            n = len(kp)
            if n == 0:
                continue
            groups.setdefault(n, []).append((idx, kp))

        for n, items in groups.items():
            names = self._feature_names(n)

            features = np.empty((len(items), n * 2), dtype=np.float32)
            for row, (_, kp) in enumerate(items):
                
                # ------------------------------------------------------------------
                # 🔄 FIXED: Convert Live Global Keypoints to Relative Box Coordinates
                # ------------------------------------------------------------------
                kp_np = np.array(kp)  # Shape: (num_keypoints, 2)
                
                # हर व्यक्ति का लोकल न्यूनतम और अधिकतम x, y ढूंढें
                xmin, ymin = np.min(kp_np, axis=0)
                xmax, ymax = np.max(kp_np, axis=0)
                
                width = xmax - xmin if (xmax - xmin) > 0 else 1.0
                height = ymax - ymin if (ymax - ymin) > 0 else 1.0
                
                for i, point in enumerate(kp):
                    # ग्लोबल वैल्यूज़ को लोकल रिलेटिव वैल्यूज़ (0 से 1) में बदलकर एरे में डालें
                    features[row, 2 * i] = (point[0] - xmin) / width
                    features[row, 2 * i + 1] = (point[1] - ymin) / height
                # ------------------------------------------------------------------

            dmatrix = xgb.DMatrix(features, feature_names=names)
            scores = self.model.predict(dmatrix)

            for (idx, _), score in zip(items, scores):
                results[idx] = float(score)

        return results

    def classify_batch(self, keypoints_list, threshold=0.55):
        scores = self.predict_batch(keypoints_list)

        output = []
        for score in scores:
            if score is None:
                output.append(
                    {"score": None, "binary": None, "is_suspicious": False}
                )
                continue

            # FIXED CORRECTION: 
            # Original Model logic: score > 0.50 means Normal (binary 1), score <= 0.50 means Cheating (binary 0).
            # New EventEngine alignment: requires binary=1 for CHEATING and binary=0 for NORMAL to calculate tracking score.
            is_cheating = score <= threshold

            output.append(
                {
                    "score": score,
                    "binary": 1 if is_cheating else 0, # Mapped correctly for New EventEngine
                    "is_suspicious": is_cheating,
                }
            )

        return output

    # ------------------------------------------------------------------
    # Backward-compatible single-person API
    # ------------------------------------------------------------------

    def predict(self, keypoints):
        return self.predict_batch([keypoints])[0]

    def classify(self, keypoints, threshold=0.55):
        return self.classify_batch([keypoints], threshold=threshold)[0]
