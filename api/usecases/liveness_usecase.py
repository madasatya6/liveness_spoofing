from api.services.liveness_service import LivenessVideoAnalyzer


class LivenessUsecase:
    """
    Business Logic:
    - memanggil service analyzer
    - menentukan label spoof/suspected/live
    - membentuk response clean untuk view
    """

    def __init__(self):
        self.analyzer = LivenessVideoAnalyzer()

    def process(self, video_base64: str):

        video_bytes = self.analyzer.decode_base64(video_base64)
        result = self.analyzer.analyze(video_bytes)

        score = result["score"]

        # ---------------------------------------------------
        # Labeling logic (tuned)
        #
        # rekomendasi threshold:
        #   < 0.40  → SPOOF
        #   0.40–0.60 → SUSPECTED
        #   > 0.60  → LIVE
        #
        # ---------------------------------------------------
        if score < 0.40:
            label = "spoof"
        elif score < 0.60:
            label = "suspected"
        else:
            label = "live"

        return {
            "score": round(score, 4),
            "label": label,
            "details": {
                "avg_blur": result["avg_blur"],
                "avg_motion": result["avg_motion"],
                "motion_std": result["motion_std"],
                "face_ratio": result["face_ratio"],
                "blink_events": result["blink_events"],
                "blink_ratio": result["blink_ratio"],
                "samples": result["samples"],
            }
        }
