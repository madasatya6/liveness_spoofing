import base64
import cv2
import numpy as np
import tempfile
import os


def _normalize(value, high):
    """Normalize 0..high → 0..1"""
    if high <= 0:
        return 0.0
    return max(0.0, min(value / high, 1.0))


class LivenessVideoAnalyzer:
    """
    Liveness Analyzer (Python 3.13 compatible)
    Features:
      ✅ Blur (sharpness)
      ✅ Motion magnitude (frame diff)
      ✅ Motion consistency
      ✅ Face detection (Haar Cascade)
      ✅ Pseudo-blink detection (brightness change)
    """

    def __init__(self):
        # OpenCV Haar face detector
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ---------------------------------------------------------------
    # Decode base64 video
    # ---------------------------------------------------------------

    def decode_base64(self, b64_string: str) -> bytes:
        if b64_string.startswith("data:"):
            b64_string = b64_string.split(",", 1)[1]
        return base64.b64decode(b64_string)

    # ---------------------------------------------------------------
    # Main analyze entrypoint
    # ---------------------------------------------------------------

    def analyze(self, video_bytes: bytes) -> dict:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)

        with open(tmp_path, "wb") as f:
            f.write(video_bytes)

        try:
            return self._score_from_video(tmp_path)
        finally:
            os.remove(tmp_path)

    # ---------------------------------------------------------------
    # Core video score
    # ---------------------------------------------------------------

    def _score_from_video(self, path, max_frames=40, frame_step=5):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Cannot read video")

        blur_vals = []
        motion_vals = []
        blink_events = 0
        face_count = 0

        prev_gray = None
        prev_eye_brightness = None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if 0 < total_frames < max_frames:
            frame_step = max(1, total_frames // max_frames)

        sampled = 0
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            if (frame_index - 1) % frame_step != 0:
                continue

            if sampled >= max_frames:
                break

            sampled += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---------------------
            # Blur (sharpness)
            # ---------------------
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            blur_vals.append(lap_var)

            # ---------------------
            # Motion
            # ---------------------
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_vals.append(float(np.mean(diff)))
            prev_gray = gray

            # ---------------------
            # Face detection
            # ---------------------
            # make face detector more reliable
            small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = self.face_detector.detectMultiScale(small, 1.1, 4)

            if len(faces) > 0:
                face_count += 1

                # take first face for blink detection
                (x, y, w, h) = faces[0]
                x *= 2; y *= 2; w *= 2; h *= 2

                eye_region = gray[y:y + h//4, x:x + w]
                if eye_region.size > 0:
                    brightness = float(np.mean(eye_region))

                    if prev_eye_brightness is not None:
                        if abs(brightness - prev_eye_brightness) > 12:
                            blink_events += 1

                    prev_eye_brightness = brightness

        cap.release()

        # ---------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------

        avg_blur = float(np.mean(blur_vals)) if blur_vals else 0
        avg_motion = float(np.mean(motion_vals)) if motion_vals else 0
        motion_std = float(np.std(motion_vals)) if motion_vals else 0
        face_ratio = face_count / sampled if sampled else 0
        blink_ratio = blink_events / sampled if sampled else 0

        # ---------------------------------------------------------------
        # Normalization
        # ---------------------------------------------------------------

        blur_norm = _normalize(avg_blur, 400)
        motion_norm = _normalize(avg_motion, 25)

        # ---------------------------------------------------------------
        # Combined Score (tuned)
        # ---------------------------------------------------------------

        score = (
            0.55 * blur_norm +
            0.25 * motion_norm +
            0.10 * face_ratio +
            0.10 * min(blink_ratio * 40, 1.0)
        )

        # extra heuristics
        if avg_blur > 420:
            score += 0.05

        if avg_blur < 180 and motion_norm > 0.4:
            score -= 0.12

        score = max(0.0, min(score, 1.0))

        return {
            "score": score,
            "avg_blur": avg_blur,
            "avg_motion": avg_motion,
            "motion_std": motion_std,
            "face_ratio": face_ratio,
            "blink_events": blink_events,
            "blink_ratio": blink_ratio,
            "samples": sampled,
        }
