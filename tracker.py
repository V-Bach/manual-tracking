from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
    drawing_styles,
)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]


@dataclass
class HandLandmarkData:
    """Structured output for detected hand landmarks and metadata."""

    handedness: str  # "Left" or "Right"
    confidence: float  # Detection / classification score [0.0, 1.0]
    landmarks: List[Dict[str, Any]] = field(default_factory=list)


class HandTracker:
    """MediaPipe HandLandmarker wrapper for real-time 21-keypoint tracking."""

    def __init__(
        self,
        model_path: str = MODEL_FILENAME,
        max_num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = model_path
        self._ensure_model_exists()

        # Configure MediaPipe HandLandmarker options
        base_options = BaseOptions(model_asset_path=self.model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        """Download model asset if not present locally."""
        if not os.path.exists(self.model_path):
            print(f"Downloading MediaPipe model asset: {self.model_path}...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("Model downloaded successfully.")

    def process(
        self, frame: cv2.Mat, timestamp_ms: int = None
    ) -> Tuple[List[HandLandmarkData], Any]:
        """
        Process a BGR image frame and extract hand landmark data.

        Args:
            frame: OpenCV BGR image matrix.
            timestamp_ms: Monotonic timestamp in milliseconds.

        Returns:
            Tuple containing:
              - List of HandLandmarkData objects for detected hands.
              - Raw HandLandmarkerResult object.
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run inference
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        detected_hands: List[HandLandmarkData] = []

        if result and result.hand_landmarks and result.handedness:
            for hand_landmarks, handedness_info in zip(
                result.hand_landmarks, result.handedness
            ):
                label = (
                    handedness_info[0].category_name
                    if handedness_info
                    else "Unknown"
                )
                score = (
                    float(handedness_info[0].score) if handedness_info else 0.0
                )

                landmarks_list = []
                for idx, lm in enumerate(hand_landmarks):
                    # Compute pixel coordinates
                    px = int(min(lm.x * width, width - 1))
                    py = int(min(lm.y * height, height - 1))

                    landmark_name = (
                        LANDMARK_NAMES[idx]
                        if idx < len(LANDMARK_NAMES)
                        else f"LANDMARK_{idx}"
                    )

                    landmarks_list.append(
                        {
                            "id": idx,
                            "name": landmark_name,
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                            "px": px,
                            "py": py,
                        }
                    )

                hand_data = HandLandmarkData(
                    handedness=label, confidence=score, landmarks=landmarks_list
                )
                detected_hands.append(hand_data)

        return detected_hands, result

    def draw_landmarks(self, frame: cv2.Mat, result: Any) -> cv2.Mat:
        """
        Draw hand skeleton and keypoint landmarks on the frame matrix.

        Args:
            frame: OpenCV image matrix (modified in-place).
            result: HandLandmarkerResult object.

        Returns:
            Annotated OpenCV frame.
        """
        if result and result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )
        return frame

    def close(self):
        """Release MediaPipe HandLandmarker resources."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()
